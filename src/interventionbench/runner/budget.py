"""Per-trajectory token budget guard.

Tracks cumulative input + output tokens (and optional USD cost) across all
LLM calls within a single trajectory. When budget exceeded, raises
BudgetExceeded which the worker catches and uses to stop the trajectory
early with a structured reason.

Usage:
    budget = TrajectoryBudget(
        max_total_tokens=int(os.environ.get("MAX_TOKENS_PER_CELL", "300000")),
        max_cost_usd=float(os.environ.get("MAX_COST_PER_CELL_USD", "1.0")),
        model_id=MODEL_ID,
    )
    response = client.chat(messages, max_tokens=2048, temperature=0.0)
    budget.add(response.usage)  # raises BudgetExceeded if limit hit
"""
from __future__ import annotations

# OpenAI pricing per 1M tokens (rough — used for cost estimate; vLLM = $0).
# Update as needed. Used only when MODEL_ID matches a known closed model.
PRICING = {
    # GPT family
    "gpt-4o":          {"in":  2.50, "out": 10.00},
    "gpt-4o-mini":     {"in":  0.15, "out":  0.60},
    "gpt-4.1":         {"in":  2.00, "out":  8.00},
    "gpt-4.1-mini":    {"in":  0.40, "out":  1.60},
    "gpt-5":           {"in":  5.00, "out": 15.00},  # placeholder
    "gpt-5-mini":      {"in":  0.25, "out":  2.00},
    # Anthropic
    "claude-sonnet-4-6":   {"in":  3.00, "out": 15.00},
    "claude-opus-4-7":     {"in": 15.00, "out": 75.00},
    "claude-haiku-4-5":    {"in":  0.80, "out":  4.00},
    # Google
    "gemini-2.0-flash":      {"in": 0.10, "out": 0.40},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-2.5-flash":      {"in": 0.30, "out": 2.50},
    "gemini-2.5-pro":        {"in": 1.25, "out": 5.00},
    # Moonshot
    "kimi-k2":               {"in": 0.40, "out": 2.00},  # placeholder
    # Local / OpenRouter / unknown → 0 cost
}


class BudgetExceeded(Exception):
    def __init__(self, reason: str, tokens: int, cost: float):
        self.reason = reason
        self.tokens = tokens
        self.cost = cost
        super().__init__(f"{reason} (tokens={tokens}, ${cost:.4f})")


def _lookup_price(model_id: str) -> tuple[float, float]:
    """Return (in_per_1m, out_per_1m). Longest-match-first so e.g.
    'gpt-4o-mini' matches the mini price, not the base 'gpt-4o' price."""
    m = (model_id or "").lower()
    for key in sorted(PRICING.keys(), key=len, reverse=True):
        if key in m:
            return PRICING[key]["in"], PRICING[key]["out"]
    return 0.0, 0.0


class TrajectoryBudget:
    def __init__(
        self,
        *,
        max_total_tokens: int = 300_000,
        max_cost_usd: float = 1.0,
        max_calls: int = 200,
        model_id: str = "",
    ):
        self.max_total_tokens = max_total_tokens
        self.max_cost_usd = max_cost_usd
        self.max_calls = max_calls
        self.in_per_1m, self.out_per_1m = _lookup_price(model_id)
        # Counters
        self.in_tokens = 0
        self.out_tokens = 0
        self.calls = 0
        self.cost = 0.0

    def add(self, usage: dict | None):
        """Add a usage record. Raises BudgetExceeded if any cap exceeded."""
        if not usage:
            usage = {}
        in_t = int(usage.get("prompt_tokens", 0) or 0)
        out_t = int(usage.get("completion_tokens", 0) or 0)
        self.in_tokens += in_t
        self.out_tokens += out_t
        self.calls += 1
        self.cost += (in_t * self.in_per_1m + out_t * self.out_per_1m) / 1_000_000.0
        total = self.in_tokens + self.out_tokens
        if total > self.max_total_tokens:
            raise BudgetExceeded(
                f"token_budget (>{self.max_total_tokens})", total, self.cost)
        if self.cost > self.max_cost_usd:
            raise BudgetExceeded(
                f"cost_budget (>${self.max_cost_usd:.2f})", total, self.cost)
        if self.calls > self.max_calls:
            raise BudgetExceeded(
                f"call_budget (>{self.max_calls})", total, self.cost)

    def snapshot(self) -> dict:
        return {
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "total_tokens": self.in_tokens + self.out_tokens,
            "calls": self.calls,
            "cost_usd": round(self.cost, 4),
        }
