"""Sensor and VLM verification evidence stored inside the episode Memory.

These measured facts and visual hypotheses never authorize movement or
constitute ground truth. Memory is the sole owner of this evidence store.
"""
import copy
import math


class EvidenceLedger:
    def __init__(self):
        self.obstacle_blocks = []
        self.surface_measurements = []
        self.verifications = []

    def blocked_forward(self, source, pose, *, reason, center,
                        place_id=None):
        """Record pose AFTER turning but BEFORE a rejected atomic forward."""
        item=dict(
            subgoal_index=source.get('subgoal'),
            place_id=place_id,
            observation_id=source['observation_id'],
            rgb_hash=source.get('rgb_hash'),
            depth_hash=source.get('depth_hash'),
            heading_deg=round(math.degrees(pose['heading'])%360,2),
            pitch_deg=pose.get('pitch_deg'),
            local_position_xz_m_est=[pose.get('x'),pose.get('z')],
            motion_epoch=pose.get('motion_epoch'),
            viewpoint_generation=pose.get('viewpoint_generation'),
            reason=reason,
            central_support_pixels=center.get('central_support_pixels'),
            nearest_along_m=center.get('nearest_along_m'),
            forward_atomic_action_executed=False,
            evidence_scope='one_current_stance_and_bearing_not_global_route_veto')
        self.obstacle_blocks.append(item)
        return copy.deepcopy(item)

    def note_surface(self, measurement):
        if measurement.get('status') not in ('MEASURED','INSUFFICIENT_DEPTH'):
            raise ValueError('invalid_surface_evidence')
        self.surface_measurements.append(copy.deepcopy(measurement))
        return copy.deepcopy(measurement)

    def note_verification(self, receipt, latest_motion_step):
        entry=dict(subgoal_index=receipt.get('subgoal_index'),
            status=receipt.get('status'),
            final_verdict=bool(receipt.get('final_verdict')),
            observation_ids=list(receipt.get('observation_ids') or []),
            missing_evidence=receipt.get('missing_evidence',''),
            latest_motion_step=latest_motion_step,
            surface_evidence_ids=[
                m.get('observation_id') for m in self.surface_measurements[-3:]],
            source='VLM_completion_assessment')
        self.verifications.append(entry)
        return copy.deepcopy(entry)

    def summary(self):
        return dict(
            latest_obstacle_blocks=copy.deepcopy(self.obstacle_blocks[-4:]),
            latest_surface_measurements=copy.deepcopy(self.surface_measurements[-3:]),
            latest_verifications=copy.deepcopy(self.verifications[-4:]))
