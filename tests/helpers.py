"""Shared test helpers: run the processing stages on scripted demonstrations."""

from __future__ import annotations

from lucius.graph import Graph
from lucius.intent import IntentEngine
from lucius.memory.episodic import EpisodicMemory
from lucius.memory.failure import FailureMemory
from lucius.provenance import DataPolicy, SourceClass
from lucius.segmentation import Segmenter, SegmentStore
from lucius.sessions import SessionKind
from lucius.skills import SkillExtractor, SkillLibrary
from lucius.taxonomy import Taxonomy, classify_task
from lucius.trajectory import TrajectoryBuilder, TrajectoryStore


class MiniPipeline:
    def __init__(self, db, sessions):
        self.db = db
        self.sessions = sessions
        self.taxonomy = Taxonomy(db)
        self.taxonomy.seed()
        self.graph = Graph(db)
        self.trajectories = TrajectoryStore(db)
        self.segment_store = SegmentStore(db)
        self.segmenter = Segmenter(self.segment_store, self.trajectories)
        self.intents = IntentEngine(db, self.taxonomy)
        self.library = SkillLibrary(db, self.graph)
        self.failures = FailureMemory(db)
        self.extractor = SkillExtractor(self.library, self.failures, self.graph)
        self.episodes = EpisodicMemory(db)

    def run(self, demo, task="simple sword blockout", source=SourceClass.USER_DEMO, kind=SessionKind.LIVE_DEMO):
        policy = DataPolicy.for_live_demo() if source == SourceClass.USER_DEMO else DataPolicy.for_external(source)
        session = self.sessions.create(user_id="u", kind=kind, policy=policy, task_text=task,
                                       task_class=classify_task(task)[0])
        self.sessions.append_events(session.id, demo.events)
        session = self.sessions.finalize(session.id)
        built = TrajectoryBuilder().build(session.id, demo.events, [], operator_log_available=True)
        self.trajectories.replace(session.id, built.steps)
        segments = self.segmenter.segment(session.id, built.steps, [], built.annotations)
        intents = {}
        for seg in segments:
            hyps = self.intents.hypotheses(seg, built.steps[seg.step_start:seg.step_end + 1], session.task_class)
            self.intents.store(seg.id, hyps)
            intents[seg.id] = self.intents.primary(seg.id)
        result = self.extractor.extract(session, built.steps, segments)
        episode = self.episodes.build(session, built.steps, segments, intents, failure_ids=result.failure_ids,
                                      skill_ids=result.skill_ids, annotations=built.annotations)
        return session, built, segments, result, episode
