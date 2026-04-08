from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


ReviewStatus = Literal[
    "approved",
    "approved_with_warnings",
    "needs_manual_review",
    "rejected_non_topology",
]

ImageType = Literal["topology", "non_topology", "unclear"]


class Device(BaseModel):
    name: str
    type: str = Field(default="unknown", description="router/switch/host/firewall/server/unknown")
    mgmt_ip: Optional[str] = None


class Interface(BaseModel):
    device: str
    name: Optional[str] = None
    kind: str = Field(default="unknown", description="physical | svi | host_nic | unknown")
    mode: Optional[str] = Field(default=None, description="access | trunk | unknown")
    allowed_vlans: Optional[str] = Field(default=None, description="e.g., '5-6' or '7,8'")
    access_vlan: Optional[str] = Field(default=None, description="e.g., '5'")
    ip: Optional[str] = None
    mask: Optional[str] = None
    vlan: Optional[str] = None
    ip_raw: Optional[str] = None
    mask_raw: Optional[str] = None
    vlan_raw: Optional[str] = None


class LinkEnd(BaseModel):
    device: str
    interface: Optional[str] = None


class Link(BaseModel):
    a: LinkEnd
    b: LinkEnd
    medium: Optional[str] = "unknown"


class Subnet(BaseModel):
    cidr: str
    members: List[LinkEnd] = Field(default_factory=list)


class TopologyExtraction(BaseModel):
    schema_version: int = 2
    devices: List[Device] = Field(default_factory=list)
    interfaces: List[Interface] = Field(default_factory=list)
    links: List[Link] = Field(default_factory=list)
    subnets: List[Subnet] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TopologyDraftArtifact(BaseModel):
    image_id: str
    experiment_id: str
    source_image: str
    extractor_model: str
    topology: TopologyExtraction = Field(default_factory=TopologyExtraction)


class TopologyImageClassificationDecision(BaseModel):
    image_type: ImageType = "unclear"
    classification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class TopologyImageClassificationArtifact(BaseModel):
    image_id: str
    experiment_id: str
    source_image: str
    classifier_model: str
    image_type: ImageType = "unclear"
    classification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class TopologyReviewDecision(BaseModel):
    image_type: ImageType = "unclear"
    review_status: ReviewStatus = "needs_manual_review"
    is_usable: bool = False
    review_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    summary: str = ""
    corrected_topology: TopologyExtraction = Field(default_factory=TopologyExtraction)


class TopologyReviewArtifact(BaseModel):
    image_id: str
    experiment_id: str
    source_image: str
    review_model: str
    image_type: ImageType = "unclear"
    review_status: ReviewStatus = "needs_manual_review"
    is_usable: bool = False
    review_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    summary: str = ""
    corrected_topology: TopologyExtraction = Field(default_factory=TopologyExtraction)


class TopologyManifestItem(BaseModel):
    image_id: str
    source_image: str
    classification_path: str
    raw_json_path: Optional[str] = None
    review_path: Optional[str] = None
    approved_json_path: Optional[str] = None
    image_type: ImageType = "unclear"
    classification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    review_status: ReviewStatus = "needs_manual_review"
    is_usable: bool = False
    review_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


class TopologyBuildManifest(BaseModel):
    schema_version: int = 1
    experiment_id: str
    experiment_label: str
    docx_path: str
    store_dir: str
    classifier_model: str
    extractor_model: str
    review_model: str
    images_total: int = 0
    non_topology_prefiltered_total: int = 0
    approved_total: int = 0
    approved_with_warnings_total: int = 0
    needs_manual_review_total: int = 0
    rejected_non_topology_total: int = 0
    items: List[TopologyManifestItem] = Field(default_factory=list)
