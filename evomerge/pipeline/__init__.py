"""evomerge.pipeline — trace-to-training-record converters."""
from evomerge.pipeline.sft import to_sft_records
from evomerge.pipeline.dpo import to_dpo_records
from evomerge.pipeline.ppo import to_ppo_records
from evomerge.pipeline.compliance_sft import compliance_to_sft_records

__all__ = [
    "compliance_to_sft_records",
    "to_dpo_records",
    "to_ppo_records",
    "to_sft_records",
]
