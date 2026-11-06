"""Source-backed evidence records shared by navigation, observation and task review.

Evidence is immutable after recording. Labels describing model hypotheses or
RGB-D estimated motion are never movement authorization or simulator truth.
"""
import copy
import math


class EvidenceLedger:
    def __init__(self):
        self.route_outcomes = []
        self.obstacle_blocks = []
        self.surface_measurements = []
        self.verifications = []
        self.completion_reviews = []

    def motion_outcome(self, candidate, action, expected_vertical='UNKNOWN'):
        """Join one explicitly chosen visual candidate to an executed forward.

        Never infer candidate identity from nearby yaw alone: the source pose
        may have changed and identical bearings need not name the same route.
        """
        if not candidate or action.get('action')!='MOVE_FORWARD':
            return None
        result=action.get('result') or {}
        motion=result.get('rgbd_motion') or {}
        translation=motion.get('translation_m') or {}
        delta=translation.get('y_up')
        uncertainty=motion.get('uncertainty_m')
        measured='UNKNOWN'
        if (motion.get('status')=='ESTIMATED' and
                isinstance(delta,(int,float)) and math.isfinite(delta) and
                isinstance(uncertainty,(int,float)) and
                math.isfinite(uncertainty) and uncertainty>=0):
            measured=('UP' if delta>uncertainty else
                      'DOWN' if delta<-uncertainty else 'LEVEL_OR_UNCERTAIN')
        prediction=candidate.get('expected_progress','UNKNOWN')
        consistency='INCONCLUSIVE'
        if prediction=='ADVANCE' and expected_vertical in ('UP','DOWN'):
            if measured in ('UP','DOWN'):
                consistency=('SUPPORTED_BY_MOTION' if measured==expected_vertical
                             else 'CONTRADICTED_BY_MOTION')
        datum=dict(candidate_id=candidate['id'],
            subgoal_index=action.get('subgoal'),
            candidate_source_observation_id=candidate.get('selected_observation_id'),
            candidate_source_heading_deg=candidate.get('absolute_heading_deg'),
            prediction=prediction,expected_vertical_motion=expected_vertical,
            actual_heading_deg=round(math.degrees(
                action.get('pose',{}).get('heading',0))%360,2),
            action_step=action['step'],
            observation_before=result.get('observation_before'),
            observation_after=result.get('observation_after'),
            vertical_change_m_est=round(delta,4) if isinstance(delta,(int,float)) else None,
            uncertainty_m_est=round(uncertainty,4)
                if isinstance(uncertainty,(int,float)) else None,
            measured_vertical_motion=measured,
            prediction_consistency=consistency,
            provenance='executed_action_RGBD_estimate_not_simulator_truth',
            movement_permission=False)
        self.route_outcomes.append(datum)
        return copy.deepcopy(datum)

    def blocked_forward(self, source, pose, *, reason, center):
        """Record pose AFTER turning but BEFORE a rejected atomic forward."""
        item=dict(
            subgoal_index=source.get('subgoal'),
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

    def note_completion_review(self, review):
        self.completion_reviews.append(copy.deepcopy(review))
        return copy.deepcopy(review)

    def summary(self):
        return dict(
            latest_route_predictions=copy.deepcopy(self.route_outcomes[-5:]),
            latest_obstacle_blocks=copy.deepcopy(self.obstacle_blocks[-4:]),
            latest_surface_measurements=copy.deepcopy(self.surface_measurements[-3:]),
            latest_verifications=copy.deepcopy(self.verifications[-4:]),
            completion_reviews=copy.deepcopy(self.completion_reviews[-3:]))
