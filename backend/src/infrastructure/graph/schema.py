ALLOWED_NODE_TYPES = {
    "Company",
    "Product",
    "Market",
    "CustomerSegment",
    "Capability",
    "Constraint",
    "Risk",
    "Goal",
    "Metric",
    "Regulation",
    "Person",
    "Location",
    "Project",
    "Event",
    "Document",
    "DocumentVersion",
}

EXTRACTABLE_NODE_TYPES = ALLOWED_NODE_TYPES - {"Document", "DocumentVersion"}

ALLOWED_RELATIONSHIPS = {
    "TARGETS",
    "OPERATES_IN",
    "REQUIRES",
    "LIMITED_BY",
    "EXPOSED_TO",
    "IMPROVES",
    "ALIGNS_WITH",
    "HAS_VERSION",
    "SUPPORTS",
    "OFFERS",
    "HAS_CAPABILITY",
    "USES",
    "DEVELOPS",
    "REGULATES",
    "LOCATED_IN",
    "MANAGES",
    "PARTICIPATES_IN",
    "AFFECTS"
}
