"""Pydantic schemas. Traceability fields are mandatory by design."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

class OccupationCandidate(BaseModel):
    onet_soc_code: str
    official_occupation_title: str
    matched_title: str = ""
    title_match_type: str = ""          # official | alternate | reported | fuzzy | explicit_code | description_suggestion
    title_source: str = ""
    mapping_status: str = ""            # from job_title_lookup
    match_score: Optional[float] = None  # title similarity only, NOT occupational relevance
    match_rationale: str = ""
    sample_tasks: list[str] = Field(default_factory=list)
    is_suggested: bool = False          # description-based suggestion; never auto-resolved

class ResolvedOccupation(BaseModel):
    onet_soc_code: str
    official_occupation_title: str
    resolution_method: str              # explicit_code | exact_title | fuzzy_confirmed
    matched_title: str = ""

class EvidenceReference(BaseModel):
    evidence_id: str                    # deterministic, derived (raw data untouched)
    onet_soc_code: str
    onet_table: str = ""
    source_file: str = ""
    onet_element_id: str = ""
    onet_element_name: str = ""
    scale_id: str = ""
    scale_name: str = ""
    original_rating: str = ""           # original Data Value as stored; "" = missing, never 0
    task_id: str = ""
    onet_release: str = "31.0"
    source_record: str = ""             # mapping/evidence locator description

class DomainRecommendation(BaseModel):
    domain_id: str
    domain_name: str
    construct_family: str
    domain_definition: str = ""         # what this domain IS (from the domain catalogue)
    relevance_explanation: str = ""     # filled by LLM or deterministic preview
    related_job_responsibilities: list[str] = Field(default_factory=list)
    mapping_status: str                 # provisional_evidence | requires_review (recommendations only)
    evidence_type: str = ""
    crosswalk_link_types: list[str] = Field(default_factory=list)
    source_rating_summary: str = ""
    preliminary_score: Optional[float] = None   # within-element percentile mean; NOT a psychometric score
    within_family_rank: Optional[int] = None    # rank ONLY within this family for this occupation
    contextual_relevance: str = ""      # job-description-based, kept separate from preliminary_score
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    confidence_explanation: str = ""
    human_review_required: bool = True
    limitations: list[str] = Field(default_factory=list)

class ConstructFamilyGroup(BaseModel):
    construct_family: str
    score_scope_note: str = "Ranked only within this family using preliminary percentiles; not comparable across families and not a validated psychometric score."
    domains: list[DomainRecommendation] = Field(default_factory=list)

class AnalysisResponse(BaseModel):
    input_job_title: str = ""
    input_job_description: str = ""
    resolution_status: str              # resolved | ambiguous_clarification_required | not_found | invalid_code
    resolved_occupation: Optional[ResolvedOccupation] = None
    occupation_candidates: list[OccupationCandidate] = Field(default_factory=list)
    analysis_status: str                # completed | clarification_required | not_analyzed | completed_with_limitations
    construct_families: list[ConstructFamilyGroup] = Field(default_factory=list)
    psychometric_domains: list[DomainRecommendation] = Field(default_factory=list)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    limitations: list[str] = Field(default_factory=list)
    llm_provider: str = "preview"
    llm_model: str = "deterministic-preview"
    is_llm_generated: bool = False

# --- LLM structured output (validated before use; refs must resolve locally) ---
class LLMExplanationItem(BaseModel):
    domain_id: str
    relevance_explanation: str
    related_job_responsibilities: list[str] = Field(default_factory=list)
    confidence_explanation: str = ""
    limitations: list[str] = Field(default_factory=list)
    evidence_reference_ids: list[str] = Field(default_factory=list)

class LLMOutput(BaseModel):
    explanations: list[LLMExplanationItem] = Field(default_factory=list)
