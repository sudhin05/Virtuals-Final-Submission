"""Contains classes and functions that handle scoring functionality and report generation."""

import json
import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import pymap3d as pm

from triage_scorer.constants import competitor_report_keys as crk
from triage_scorer.constants import ground_truth_keys as gtk


# Golden window is defaulted to 15 minutes
golden_window: int = 900


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ScorableField:
    # Golden windows need to depend on the submission time of a report, not the assessment time
    def __init__(
        self,
        assessment_time: float,
        report_time: float,
        lon: float,
        lat: float,
        alt: float,
        value,
    ):
        self.assessment_time = assessment_time
        self.report_time = report_time
        self.lon = lon
        self.lat = lat
        self.alt = alt
        self.value = value

    def __repr__(self):
        return f"{self.value} @ t={self.assessment_time} & loc=({self.lon},{self.lat},{self.alt})"


class CasualtyAssessmentTracker:
    def __init__(self):
        self._aggregate = dict()
        self._reports = []

    def _update_field(self, key_chain: list[str], field: ScorableField):
        if not key_chain:
            return

        data = self._aggregate

        # Traverse down the key tree to the last non-leaf property
        for key in key_chain[:-1]:
            if key not in data:
                data[key] = dict()
            data = data[key]

        data[key_chain[-1]] = field

    def track_report(self, validated_report: dict, report_time: float):
        self._reports.append(validated_report)

        # These keys are guaranteed to be available in a validated report
        a_time = validated_report[crk.ASSESSMENT_TIME]
        lon, lat, alt = (
            validated_report[crk.LOCATION][crk.LON],
            validated_report[crk.LOCATION][crk.LAT],
            validated_report[crk.LOCATION][crk.ALT],
        )

        # traverse the report and create a ScorableField for each terminal/leaf property
        unscored_keys = [
            crk.OBSERVATION_START,
            crk.OBSERVATION_END,
            crk.ASSESSMENT_TIME,
            crk.LOCATION,
            crk.CASUALTY_ID,
            crk.DRONE_ID,
        ]
        for health_key in [key for key in validated_report if key not in unscored_keys]:
            if isinstance(validated_report[health_key], dict):
                self._recursive_track_field(
                    [health_key],
                    validated_report[health_key],
                    assessment_time=a_time,
                    report_time=report_time,
                    lon=lon,
                    lat=lat,
                    alt=alt,
                )
            else:
                scorable = ScorableField(
                    assessment_time=a_time,
                    report_time=report_time,
                    lon=lon,
                    lat=lat,
                    alt=alt,
                    value=validated_report[health_key],
                )
                self._update_field([health_key], scorable)

    def _recursive_track_field(
        self,
        key_chain: list[str],
        subreport: dict,
        assessment_time: float,
        report_time: float,
        lon: float,
        lat: float,
        alt: float,
    ):
        for key in subreport.keys():
            if isinstance(subreport[key], dict):
                self._recursive_track_field(
                    [*key_chain, key],
                    subreport[key],
                    assessment_time=assessment_time,
                    report_time=report_time,
                    lon=lon,
                    lat=lat,
                    alt=alt,
                )
            else:
                scorable = ScorableField(
                    assessment_time=assessment_time,
                    report_time=report_time,
                    lon=lon,
                    lat=lat,
                    alt=alt,
                    value=subreport[key],
                )
                self._update_field([*key_chain, key], scorable)


def generate_final_report(
    final_report_filepath: str,
    cats: defaultdict[str, CasualtyAssessmentTracker],
    gts: list[tuple[tuple[float, float, float], pd.DataFrame]],
    runtime: int
):
    """Generates the final report which includes the total score and the list of assessments for each CasualtyAssessmentTracker.

    Args:
        final_report_filepath (str): The filepath where the final report should be created and stored
        cats (defaultdict[str, CasualtyAssessmentTracker]): Collection of mappings of casualty IDs to CasualtyAssessmentTrackers
        gts (list[tuple[tuple[float, float, float], pd.DataFrame]]): Mapping of casualty locations to ground truth data
    """
    final_report: dict = {}

    final_score, final_casualty_assessments = generate_final_score_and_casualty_assessments(
        cats=cats,
        gts=gts,
    )
    final_report["runtime"] = runtime
    final_report["golden_window"] = golden_window
    final_report["final_score"] = final_score
    final_report["casualty_assessments"] = final_casualty_assessments

    with open(final_report_filepath, "w") as file:
        json.dump(final_report, file, indent=4)


def generate_final_score_and_casualty_assessments(
    cats: defaultdict[str, CasualtyAssessmentTracker],
    gts: tuple[tuple[float, float, float], pd.DataFrame],
) -> tuple[float, dict[str, Any]]:
    """Computes the final score and the casualty assessments for each casualty ID.

    Args:
        cats (defaultdict[str, CasualtyAssessmentTracker]): Collection of mappings of casualty IDs to CasualtyAssessmentTrackers
        gts (list[tuple[tuple[float, float, float], pd.DataFrame]]): Mapping of casualty locations to ground truth data

    Returns:
        tuple[float, dict[str, Any]]: The final score and final casualty assessments
    """
    final_score: float = 0.0
    final_casualty_assessments: dict[str, Any] = {}
    gts_used: set[tuple[float, float, float]] = set()

    for casualty_id in cats:
        assessment: dict = {}
        assessment["reports"] = []

        # Add the reports to the assessment
        for report in cats[casualty_id]._reports:
            assessment["reports"].append(format_reports(report))

        # Add the formatted aggregate to the assessment
        assessment["aggregate"] = format_aggregate(cats[casualty_id]._aggregate)

        # Compute the score of the aggregate and ground truths used
        aggregate_score, casualty_gts_used = score_casualty(cat=cats[casualty_id], gts=gts, gts_used=gts_used)

        for gt_loc in casualty_gts_used:
            gts_used.add(gt_loc)

        # Add the score to the assessment
        assessment["score"] = aggregate_score

        # Map the assessment to the casualty ID
        final_casualty_assessments[casualty_id] = assessment
        final_score += aggregate_score

    return final_score, final_casualty_assessments


def format_reports(report: dict) -> dict:
    """Formats a report to match the example_report.json.

    Args:
        report (dict): The report that will be referenced to generate a new, formatted report

    Returns:
        dict: The formatted report
    """
    formatted_report: dict = {}
    formatted_report[crk.CASUALTY_ID] = report[crk.CASUALTY_ID]
    formatted_report[crk.OBSERVATION_START] = report[crk.OBSERVATION_START]
    formatted_report[crk.OBSERVATION_END] = report[crk.OBSERVATION_END]
    formatted_report[crk.ASSESSMENT_TIME] = report[crk.ASSESSMENT_TIME]
    formatted_report[crk.LOCATION] = report[crk.LOCATION]

    if crk.VITALS in report:
        formatted_report[crk.VITALS] = {}

        if crk.HEART_RATE in report[crk.VITALS]:
            formatted_report[crk.VITALS][crk.HEART_RATE] = report[crk.VITALS][crk.HEART_RATE]

        if crk.RESPIRATION_RATE in report[crk.VITALS]:
            formatted_report[crk.VITALS][crk.RESPIRATION_RATE] = report[crk.VITALS][crk.RESPIRATION_RATE]

    if crk.INJURIES in report:
        formatted_report[crk.INJURIES] = {}

        if crk.SEVERE_HEMORRHAGE in report[crk.INJURIES]:
            formatted_report[crk.INJURIES][crk.SEVERE_HEMORRHAGE] = report[crk.INJURIES][crk.SEVERE_HEMORRHAGE]

        if crk.RESPIRATORY_DISTRESS in report[crk.INJURIES]:
            formatted_report[crk.INJURIES][crk.RESPIRATORY_DISTRESS] = report[crk.INJURIES][crk.RESPIRATORY_DISTRESS]

        if crk.TRAUMA in report[crk.INJURIES]:
            formatted_report[crk.INJURIES][crk.TRAUMA] = report[crk.INJURIES][crk.TRAUMA]

        if crk.ALERTNESS in report[crk.INJURIES]:
            formatted_report[crk.INJURIES][crk.ALERTNESS] = report[crk.INJURIES][crk.ALERTNESS]

    return formatted_report


def format_aggregate(aggregate: dict) -> dict:
    """Extracts the values from the ScorableField objects stored in the aggregate and generates a new dict object.

    Args:
        aggregate (dict): The aggregate object of the CasualtyAssessmentTracker in which values will be extracted from

    Returns:
        dict: The formatted dict containing the extracted values from the ScorableField objects stored in the aggregate
    """
    formatted_aggregate = {}

    if crk.VITALS in aggregate:
        formatted_aggregate[crk.VITALS] = {}

        if crk.HEART_RATE in aggregate[crk.VITALS]:
            formatted_aggregate[crk.VITALS][crk.HEART_RATE] = aggregate[crk.VITALS][crk.HEART_RATE].value

        if crk.RESPIRATION_RATE in aggregate[crk.VITALS]:
            formatted_aggregate[crk.VITALS][crk.RESPIRATION_RATE] = aggregate[crk.VITALS][crk.RESPIRATION_RATE].value

    if crk.INJURIES in aggregate:
        formatted_aggregate[crk.INJURIES] = {}

        if crk.SEVERE_HEMORRHAGE in aggregate[crk.INJURIES]:
            formatted_aggregate[crk.INJURIES][crk.SEVERE_HEMORRHAGE] = aggregate[crk.INJURIES][
                crk.SEVERE_HEMORRHAGE
            ].value

        if crk.RESPIRATORY_DISTRESS in aggregate[crk.INJURIES]:
            formatted_aggregate[crk.INJURIES][crk.RESPIRATORY_DISTRESS] = aggregate[crk.INJURIES][
                crk.RESPIRATORY_DISTRESS
            ].value

        if crk.TRAUMA in aggregate[crk.INJURIES]:
            formatted_aggregate[crk.INJURIES][crk.TRAUMA] = {}

            if crk.HEAD in aggregate[crk.INJURIES][crk.TRAUMA]:
                formatted_aggregate[crk.INJURIES][crk.TRAUMA][crk.HEAD] = aggregate[crk.INJURIES][crk.TRAUMA][
                    crk.HEAD
                ].value

            if crk.TORSO in aggregate[crk.INJURIES][crk.TRAUMA]:
                formatted_aggregate[crk.INJURIES][crk.TRAUMA][crk.TORSO] = aggregate[crk.INJURIES][crk.TRAUMA][
                    crk.TORSO
                ].value

            if crk.UPPER_EXTREMITY in aggregate[crk.INJURIES][crk.TRAUMA]:
                formatted_aggregate[crk.INJURIES][crk.TRAUMA][crk.UPPER_EXTREMITY] = aggregate[crk.INJURIES][
                    crk.TRAUMA
                ][crk.UPPER_EXTREMITY].value

            if crk.LOWER_EXTREMITY in aggregate[crk.INJURIES][crk.TRAUMA]:
                formatted_aggregate[crk.INJURIES][crk.TRAUMA][crk.LOWER_EXTREMITY] = aggregate[crk.INJURIES][
                    crk.TRAUMA
                ][crk.LOWER_EXTREMITY].value

        if crk.ALERTNESS in aggregate[crk.INJURIES]:
            formatted_aggregate[crk.INJURIES][crk.ALERTNESS] = {}

            if crk.OCULAR in aggregate[crk.INJURIES][crk.ALERTNESS]:
                formatted_aggregate[crk.INJURIES][crk.ALERTNESS][crk.OCULAR] = aggregate[crk.INJURIES][crk.ALERTNESS][
                    crk.OCULAR
                ].value

            if crk.VERBAL in aggregate[crk.INJURIES][crk.ALERTNESS]:
                formatted_aggregate[crk.INJURIES][crk.ALERTNESS][crk.VERBAL] = aggregate[crk.INJURIES][crk.ALERTNESS][
                    crk.VERBAL
                ].value

            if crk.MOTOR in aggregate[crk.INJURIES][crk.ALERTNESS]:
                formatted_aggregate[crk.INJURIES][crk.ALERTNESS][crk.MOTOR] = aggregate[crk.INJURIES][crk.ALERTNESS][
                    crk.MOTOR
                ].value

    return formatted_aggregate


def score_casualty(
    cat: CasualtyAssessmentTracker,
    gts: list[tuple[float, float, float], pd.DataFrame],
    gts_used: set[tuple[float, float, float]],
) -> tuple[float, set[tuple[float, float, float]]]:
    """Score the aggregate and compile a list of ground truth locations.

    Args:
        cat (CasualtyAssessmentTracker): CasualtyAssessmentTracker for a specified casualty ID
        gts (list[tuple[tuple[float, float, float], pd.DataFrame]]): Mapping of casualty locations to ground truth data
        gts_used (set[tuple[float, float, float]]): The set of ground truths used when scoring aggregates

    Returns:
        tuple[float, set[tuple[float, float, float]]]: The aggregate score and the set of used ground truth locations
    """
    aggregate_score: float = 0.0

    vitals_score, vitals_gts_used = score_vitals(cat, gts, gts_used)
    hemdis_score, hemdis_gts_used = score_hemorrhage_and_distress(cat, gts, gts_used)
    trauma_score, trauma_gts_used = score_trauma(cat, gts, gts_used)
    alertness_score, alertness_gts_used = score_alertness(cat, gts, gts_used)

    aggregate_score = vitals_score + hemdis_score + trauma_score + alertness_score
    casualty_gts_used: set[tuple[float, float, float]] = set.union(
        vitals_gts_used, hemdis_gts_used, trauma_gts_used, alertness_gts_used
    )

    return aggregate_score, casualty_gts_used


# NOTE: At the moment this function assumes heart rate and respiratory rate are piecewise
#       constant on a given time interval. This function will need to be updated to calculate
#       the ground truth for these values if we decide to linearly interpolate hr/rr or use
#       some other function to model them.
def score_vitals(
    cat: CasualtyAssessmentTracker,
    ground_truths: list[tuple[tuple[float, float, float], pd.DataFrame]],
    gts_used: set[tuple[float, float, float]],
) -> tuple[float, set[tuple[float, float, float]]]:
    """Scores the vitals ScorableFields by finding the nearest ground truth location and data and comparing it to the ScorableField in the aggregate.

    Args:
        cat (CasualtyAssessmentTracker): CasualtyAssessmentTracker for a specified casualty ID
        gts (list[tuple[tuple[float, float, float], pd.DataFrame]]): Mapping of casualty locations to ground truth data
        gts_used (set[tuple[float, float, float]]): The set of ground truths used when scoring aggregates

    Returns:
        tuple[float, set[tuple[float, float, float]]]: The score of the vitals ScorableFields and set of ground truths used
    """
    score: float = 0.0
    vitals_gts_used: set[tuple[float, float, float]] = set()

    # Tolerances for vitals - these are notional
    hr_tolerance = 5  # beats/min
    rr_tolerance = 3  # respirations/min

    if crk.VITALS in cat._aggregate:
        if crk.HEART_RATE in cat._aggregate[crk.VITALS]:
            field: ScorableField = cat._aggregate[crk.VITALS][crk.HEART_RATE]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                score += 0.0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    score += 0.0
                else:
                    score += (1 if abs(row[gtk.HEART_RATE] - field.value) < hr_tolerance else 0) * (
                        1.5 if field.report_time <= golden_window else 1
                    )
            vitals_gts_used.add(location)

        if crk.RESPIRATION_RATE in cat._aggregate[crk.VITALS]:
            field: ScorableField = cat._aggregate[crk.VITALS].get(crk.RESPIRATION_RATE)
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                score += 0.0
            else:
                time_window: float = 60.0
                row = get_time_proximal_row(gt, field, time_window)
                if row is None:
                    score += 0.0
                else:
                    avg_rr: float = compute_avg_rr(row, field.assessment_time, time_window)
                    score += (1 if abs(avg_rr - field.value) < rr_tolerance else 0) * (
                        1.5 if field.report_time <= golden_window else 1
                    )
            vitals_gts_used.add(location)

    return score, vitals_gts_used


def score_hemorrhage_and_distress(
    cat: CasualtyAssessmentTracker,
    ground_truths: list[tuple[tuple[float, float, float], pd.DataFrame]],
    gts_used: set[tuple[float, float, float]],
) -> tuple[float, set[tuple[float, float, float]]]:
    """Scores the hemorrhage and respiratory distress ScorableFields by finding the nearest ground truth location and data and comparing it to the ScorableField in the aggregate.

    Args:
        cat (CasualtyAssessmentTracker): CasualtyAssessmentTracker for a specified casualty ID
        gts (list[tuple[tuple[float, float, float], pd.DataFrame]]): Mapping of casualty locations to ground truth data
        gts_used (set[tuple[float, float, float]]): The set of ground truths used when scoring aggregates

    Returns:
        tuple[float, set[tuple[float, float, float]]]: The score of the hemorrhage and respiratory distress ScorableFields and set of ground truths used
    """
    score: float = 0.0
    hem_dist_gts_used: set[tuple[float, float, float]] = set()

    if crk.INJURIES in cat._aggregate:
        if crk.SEVERE_HEMORRHAGE in cat._aggregate[crk.INJURIES]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.SEVERE_HEMORRHAGE]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                score += 0.0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    score += 0.0
                else:
                    gt_severe_hemorrhage_as_bool = None
                    if row[gtk.HA_SEVERE_HEMORRHAGE].lower() == "absent":
                        gt_severe_hemorrhage_as_bool = False
                    elif row[gtk.HA_SEVERE_HEMORRHAGE].lower() == "present":
                        gt_severe_hemorrhage_as_bool = True

                    score += (2 if gt_severe_hemorrhage_as_bool == field.value else 0) * (
                        2 if field.report_time <= golden_window else 1
                    )
            hem_dist_gts_used.add(location)

        if crk.RESPIRATORY_DISTRESS in cat._aggregate[crk.INJURIES]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.RESPIRATORY_DISTRESS]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                score += 0.0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    score += 0.0
                else:
                    gt_respiratory_distress_as_bool = None
                    if row[gtk.HA_RESPIRATORY_DISTRESS].lower() == "absent":
                        gt_respiratory_distress_as_bool = False
                    elif row[gtk.HA_RESPIRATORY_DISTRESS].lower() == "present":
                        gt_respiratory_distress_as_bool = True

                    score += (2 if gt_respiratory_distress_as_bool == field.value else 0) * (
                        2 if field.report_time <= golden_window else 1
                    )

            hem_dist_gts_used.add(location)

    return score, hem_dist_gts_used


def score_trauma(
    cat: CasualtyAssessmentTracker,
    ground_truths: list[tuple[tuple[float, float, float], pd.DataFrame]],
    gts_used: set[tuple[float, float, float]],
) -> tuple[float, set[tuple[float, float, float]]]:
    """Scores the trauma ScorableFields by finding the nearest ground truth location and data and comparing it to the ScorableField in the aggregate.

    Args:
        cat (CasualtyAssessmentTracker): CasualtyAssessmentTracker for a specified casualty ID
        gts (list[tuple[tuple[float, float, float], pd.DataFrame]]): Mapping of casualty locations to ground truth data
        gts_used (set[tuple[float, float, float]]): The set of ground truths used when scoring aggregates

    Returns:
        tuple[float, set[tuple[float, float, float]]]: The score of the trauma ScorableFields and set of ground truths used
    """
    trauma_correct_count: int = 0
    trauma_gts_used: set[tuple[float, float, float]] = set()

    if crk.INJURIES in cat._aggregate and crk.TRAUMA in cat._aggregate[crk.INJURIES]:
        if crk.HEAD in cat._aggregate[crk.INJURIES][crk.TRAUMA]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.TRAUMA][crk.HEAD]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                trauma_correct_count += 0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    trauma_correct_count += 0
                else:
                    trauma_correct_count += 1 if row[gtk.HA_HEAD_TRAUMA].lower() == field.value.lower() else 0
            trauma_gts_used.add(location)

        if crk.TORSO in cat._aggregate[crk.INJURIES][crk.TRAUMA]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.TRAUMA][crk.TORSO]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                trauma_correct_count += 0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    trauma_correct_count += 0
                else:
                    trauma_correct_count += 1 if row[gtk.HA_TORSO_TRAUMA].lower() == field.value.lower() else 0
            trauma_gts_used.add(location)

        if crk.UPPER_EXTREMITY in cat._aggregate[crk.INJURIES][crk.TRAUMA]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.TRAUMA][crk.UPPER_EXTREMITY]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                trauma_correct_count += 0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    trauma_correct_count += 0
                else:
                    trauma_correct_count += 1 if row[gtk.HA_UPPER_EXTREMITY_TRAUMA].lower() == field.value.lower() else 0
            trauma_gts_used.add(location)

        if crk.LOWER_EXTREMITY in cat._aggregate[crk.INJURIES][crk.TRAUMA]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.TRAUMA][crk.LOWER_EXTREMITY]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                trauma_correct_count += 0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    trauma_correct_count += 0
                else:
                    trauma_correct_count += 1 if row[gtk.HA_LOWER_EXTREMITY_TRAUMA].lower() == field.value.lower() else 0
            trauma_gts_used.add(location)

    if trauma_correct_count == 4:
        return 2.0, trauma_gts_used
    elif trauma_correct_count >= 2:
        return 1.0, trauma_gts_used
    else:
        return 0.0, trauma_gts_used


def score_alertness(
    cat: CasualtyAssessmentTracker,
    ground_truths: list[tuple[tuple[float, float, float], pd.DataFrame]],
    gts_used: set[tuple[float, float, float]],
) -> tuple[float, set[tuple[float, float, float]]]:
    """Scores the alertness ScorableFields by finding the nearest ground truth location and data and comparing it to the ScorableField in the aggregate.

    Args:
        cat (CasualtyAssessmentTracker): CasualtyAssessmentTracker for a specified casualty ID
        gts (list[tuple[tuple[float, float, float], pd.DataFrame]]): Mapping of casualty locations to ground truth data
        gts_used (set[tuple[float, float, float]]): The set of ground truths used when scoring aggregates

    Returns:
        tuple[float, set[tuple[float, float, float]]]: The score of the alertness ScorableFields and set of ground truths used
    """
    alertness_correct_count: int = 0
    alertness_gts_used: set[tuple[float, float, float]] = set()

    if crk.INJURIES in cat._aggregate and crk.ALERTNESS in cat._aggregate[crk.INJURIES]:
        if crk.OCULAR in cat._aggregate[crk.INJURIES][crk.ALERTNESS]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.ALERTNESS][crk.OCULAR]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                alertness_correct_count += 0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    alertness_correct_count += 0
                else:
                    alertness_correct_count += 1 if row[gtk.HA_OCULAR_ALERTNESS].lower() == field.value.lower() else 0
            alertness_gts_used.add(location)

        if crk.VERBAL in cat._aggregate[crk.INJURIES][crk.ALERTNESS]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.ALERTNESS][crk.VERBAL]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                alertness_correct_count += 0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    alertness_correct_count += 0
                else:
                    alertness_correct_count += 1 if row[gtk.HA_VERBAL_ALERTNESS].lower() == field.value.lower() else 0
            alertness_gts_used.add(location)

        if crk.MOTOR in cat._aggregate[crk.INJURIES][crk.ALERTNESS]:
            field: ScorableField = cat._aggregate[crk.INJURIES][crk.ALERTNESS][crk.MOTOR]
            location, gt = get_space_proximal_gt(ground_truths, field)

            if location in gts_used:
                alertness_correct_count += 0
            else:
                row = get_time_proximal_row(gt, field)
                if row is None:
                    alertness_correct_count += 0
                else:
                    alertness_correct_count += 1 if row[gtk.HA_MOTOR_ALERTNESS].lower() == field.value.lower() else 0
            alertness_gts_used.add(location)

    if alertness_correct_count == 3:
        return 2.0, alertness_gts_used
    elif alertness_correct_count == 2:
        return 1.0, alertness_gts_used
    else:
        return 0.0, alertness_gts_used


def get_space_proximal_gt(
    ground_truths: list[tuple[tuple[float, float, float], pd.DataFrame]],
    sf: ScorableField,
) -> tuple[tuple[float, float, float], pd.DataFrame]:
    location: tuple[float, float, float] | None = None
    min_dist = None
    ret_df = None

    sf_x, sf_y, sf_z = pm.geodetic2ecef(lon=sf.lon, lat=sf.lat, alt=sf.alt, deg=True)

    for (lon, lat, alt), df in ground_truths:
        x, y, z = pm.geodetic2ecef(lon=lon, lat=lat, alt=alt, deg=True)
        dist = np.linalg.norm(np.array((x, y, z)) - np.array((sf_x, sf_y, sf_z)))
        if min_dist is None or dist < min_dist:
            location = (lon, lat, alt)
            min_dist = dist
            ret_df = df

    return location, ret_df


def get_time_proximal_row(
    ground_truth: pd.DataFrame, sf: ScorableField, time_window: Optional[float] = None
) -> pd.Series | pd.DataFrame | None:
    """Gets a row as a Series or rows as a DataFrame from a ground truth based on the time window. If the time window extends behind the available rows, then the earliest becomes the lower extent of the time window.

    Args:
        ground_truth (pd.DataFrame): Contains ground truth data
        sf (ScorableField): Accesses the assessment time
        time_window (Optional[float]): The extent of time in which to capture rows, starting at the assessment time and ending at the assessment time - time window

    Returns:
        pd.Series, pd.DataFrame, None: If no time window is provided, a Series, if a valid time window is provided, a DataFrame. If the assessment time is earlier than any available rows, then this assessment is considered invalid and None is returned.
    """
    # Get the row closest in time to sf.assessment_time
    end_idx = abs(ground_truth[gtk.TIME] - sf.assessment_time).idxmin()

    if ground_truth.iloc[end_idx][gtk.TIME] > sf.assessment_time:
        # The closest row was in the future, so step back to the "current" row
        end_idx -= 1

    if end_idx == -1:
        # sf.assessment_time was before any ground truth times
        return None

    if time_window is None:
        return ground_truth.iloc[end_idx]

    start_idx = end_idx

    while ground_truth.iloc[start_idx][gtk.TIME] > round(sf.assessment_time - time_window, 2):
        start_idx -= 1

        if start_idx == 0:
            break

    return ground_truth.iloc[start_idx : end_idx + 1]


def compute_avg_rr(df: pd.DataFrame, assessment_time: float, time_window: float) -> float:
    """Compute the average of the respiratory rates over the time window in seconds.

    Args:
        df (pd.DataFrame): DataFrame containing only the respiratory rates
        assessment_time (float): The time an assessment occurred, used here to compute the weight of each interval of respiration rates
        time_window(float): The divisor of the equation in which the average rate of respiration is computed

    Returns:
        float: The average respiratory rate
    """
    condensed_series = df.apply(lambda row: (row["Time"], row["RespirationRate"]), axis=1)
    reversed_series = condensed_series[::-1]

    sum_of_rates: float = 0.0
    curr_time = assessment_time
    stop_time = max(assessment_time - time_window, 0)

    for _, (time, respiration_rate) in reversed_series.items():
        time_step = curr_time - max(time, stop_time)
        if time_step < 0:  # `reversed_series` goes further back in time than necessary, so we're done
            break
        sum_of_rates += time_step * respiration_rate
        curr_time = time

    return sum_of_rates / (assessment_time - stop_time)
