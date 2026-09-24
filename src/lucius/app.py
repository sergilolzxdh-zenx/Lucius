"""Composition root: builds every subsystem from one configuration.

Subsystems only know the collaborators passed to them; this module is the one place where the
whole architecture is assembled (and therefore the place to read to understand it).
"""

from __future__ import annotations

from pathlib import Path

from lucius.blender.bridge import BlenderBridge
from lucius.config import LuciusConfig, load_config
from lucius.corrections import CorrectionStore
from lucius.errors import BlenderUnavailable
from lucius.evaluation import Evaluator
from lucius.events import EventBus, EventType
from lucius.executor import BridgeBackend, ExecutionEngine, InteractiveHumanChannel
from lucius.graph import Graph
from lucius.intent import IntentEngine
from lucius.logging_setup import configure_logging, get_logger
from lucius.memory.episodic import EpisodicMemory
from lucius.memory.failure import FailureMemory
from lucius.memory.preferences import WorkflowPreferences
from lucius.memory.semantic import SemanticMemory
from lucius.planner import Planner
from lucius.processing import ProcessingPipeline, on_session_end
from lucius.providers import Providers, build_providers
from lucius.recorder import AgentActionLedger, DemonstrationRecorder, RecorderSources, detect_sources
from lucius.recorder.recorder import recover_interrupted_sessions
from lucius.retrieval import HybridRetriever, VectorIndex
from lucius.segmentation import SegmentEditor, Segmenter, SegmentRefiner, SegmentStore, SignalConfig
from lucius.sessions import SessionStore
from lucius.skills import SkillExtractor, SkillLibrary
from lucius.skills.seeds import seed_system_skills
from lucius.storage import Database, FrameStore
from lucius.taxonomy import Taxonomy
from lucius.trajectory import TrajectoryStore

log = get_logger("app")


class Lucius:
    def __init__(self, config: LuciusConfig | None = None, *, data_dir: str | Path | None = None,
                 providers: Providers | None = None, background_processing: bool = True) -> None:
        configure_logging()
        self.config = config or load_config(data_dir)
        self.config.ensure_dirs()
        self.background_processing = background_processing
        cfg = self.config
        self.db = Database(cfg.db_path)
        self.frames = FrameStore(cfg.frames_dir, image_format=cfg.recording.image_format,
                                 jpeg_quality=cfg.recording.jpeg_quality, max_width=cfg.recording.max_frame_width)
        self.bus = EventBus(self.db)
        self.sessions = SessionStore(self.db, self.frames)
        self.sessions.ensure_user(cfg.user_id)
        self.taxonomy = Taxonomy(self.db)
        self.taxonomy.seed()
        self.graph = Graph(self.db)
        self.trajectories = TrajectoryStore(self.db)
        self.segments = SegmentStore(self.db)
        self.segmenter = Segmenter(self.segments, self.trajectories, self.bus, SignalConfig(
            pause_threshold_s=cfg.processing.pause_threshold_s,
            navigation_min_steps=cfg.processing.navigation_min_steps,
            visual_change_threshold=cfg.processing.visual_change_threshold),
            representative_frames=cfg.processing.representative_frames_per_segment)
        self.segment_editor = SegmentEditor(self.db, self.segments, self.trajectories, self.taxonomy, self.bus,
                                            cfg.user_id)
        self.intents = IntentEngine(self.db, self.taxonomy, self.bus)
        self.library = SkillLibrary(self.db, self.graph, self.bus)
        self.failures = FailureMemory(self.db, self.bus)
        self.extractor = SkillExtractor(self.library, self.failures, self.graph, self.bus)
        self.episodes = EpisodicMemory(self.db, self.bus)
        self.semantic = SemanticMemory(self.db, self.bus)
        self.preferences = WorkflowPreferences(self.db, cfg.user_id)
        self.corrections = CorrectionStore(self.db, self.graph, self.bus)
        if providers is None:
            providers, notes = build_providers(cfg.providers, self.db)
        else:
            notes = {}
        self.providers = providers
        self.provider_notes = notes
        self.refiner = SegmentRefiner(providers, self.segments, self.frames, self.taxonomy,
                                      max_images=cfg.providers.max_images_per_call)
        self.index = VectorIndex(self.db, providers.embeddings)
        self.retriever = HybridRetriever(self.db, self.index, self.library, self.failures, self.semantic,
                                         self.episodes, self.graph, self.preferences, self.bus)
        self.planner = Planner(self.library, self.failures, self.graph, self.bus)
        self.evaluator = Evaluator(self.db, providers, self.bus)
        self.pipeline = ProcessingPipeline(self)
        self.engine = ExecutionEngine(
            db=self.db, sessions=self.sessions, library=self.library, failures=self.failures, retriever=self.retriever,
            planner=self.planner, evaluator=self.evaluator, corrections=self.corrections, preferences=self.preferences,
            safety=cfg.safety, bus=self.bus, user_id=cfg.user_id,
            on_session_complete=self.pipeline.submit if background_processing else self.pipeline.process)
        self.bus.subscribe(EventType.SESSION_ENDED, on_session_end(self.pipeline, background=background_processing))
        self.ledger = AgentActionLedger()
        self.human = InteractiveHumanChannel()
        self._recorder: DemonstrationRecorder | None = None
        self._backend: BridgeBackend | None = None
        self._headless = None
        seed_system_skills(self.library)
        recovered = recover_interrupted_sessions(self.sessions, cfg.journal_dir, self.bus)
        for session_id in recovered:
            log.info("recovered interrupted recording %s", session_id)
        from lucius.ingestion import IngestionService

        self.ingestion = IngestionService(self)
        from lucius.practice import PracticeEngine

        self.practice = PracticeEngine(self)
        from lucius.dataset import DatasetService

        self.datasets = DatasetService(self)
        from lucius.benchmarks import BenchmarkService

        self.benchmarks = BenchmarkService(self)

    # -- capture -------------------------------------------------------------------------------------
    def recorder(self, sources: RecorderSources | None = None) -> DemonstrationRecorder:
        if self._recorder is None or sources is not None:
            if sources is None:
                sources = detect_sources(self.config)
            self._recorder = DemonstrationRecorder(self.config, self.sessions, self.bus, sources, ledger=self.ledger)
        return self._recorder

    # -- execution backends ------------------------------------------------------------------------------
    def live_backend(self) -> BridgeBackend:
        """Backend attached to the user's running Blender (via the add-on's discovery file)."""
        if self._backend is None or not self._backend.bridge.connected:
            bridge = BlenderBridge.from_config(self.config.blender)
            bridge.connect()
            self._backend = BridgeBackend(bridge)
        return self._backend

    def gui_backend(self) -> BridgeBackend:
        """The user's running Blender, driven by keyboard and mouse (observed and verified through the add-on)."""
        from lucius.executor.gui import GuiActuator, GuiBackend, create_injector, create_locator

        bridge = self.live_backend().bridge
        cfg = self.config.gui
        pid = bridge.request("gui_layout").get("pid")
        actuator = GuiActuator(create_injector(), create_locator(pid), self.ledger, event_delay_s=cfg.event_delay_s,
                               interference_px=cfg.interference_px, activate_window=cfg.activate_window)
        return GuiBackend(bridge, actuator, fallback_to_bridge=cfg.fallback_to_bridge,
                          verify_timeout_s=cfg.verify_timeout_s)

    def headless_backend(self) -> BridgeBackend:
        """A private headless Blender for practice, validation and benchmarks."""
        from lucius.blender.headless import HeadlessBlender

        if self._headless is None or self._headless.bridge is None or not self._headless.bridge.connected:
            self._headless = HeadlessBlender(allowed_save_dirs=[str(self.config.exports_dir)],
                                             allowed_read_dirs=[str(self.config.media_dir)])
            self._headless.start()
        return BridgeBackend(self._headless.bridge)

    def backend(self, prefer: str = "live") -> BridgeBackend:
        if prefer == "live":
            try:
                return self.live_backend()
            except BlenderUnavailable:
                log.info("no live Blender bridge; using headless Blender")
        return self.headless_backend()

    def close(self) -> None:
        self.pipeline.shutdown()
        if self._recorder is not None and self._recorder.recording:
            self._recorder.stop()
        if self._headless is not None:
            self._headless.stop()
        if self._backend is not None:
            self._backend.bridge.close()
        self.db.close()
