import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None

def _estimate_tokens(text: str) -> int:
    if _enc is not None:
        return max(1, len(_enc.encode(text, disallowed_special=())))
    return max(1, len(text) // 4)

@dataclass
class PromptFragment:
    name: str
    content: str
    priority: int  # Higher is more important (100 = base persona, 10 = hints)

class DynamicPromptBuilder:
    """Dynamically constructs the system prompt based on token budgets and priority."""
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.fragments: Dict[str, PromptFragment] = {}
        
    def set_budget(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        
    def add_fragment(self, name: str, content: str, priority: int = 50) -> None:
        if not content.strip():
            return
        self.fragments[name] = PromptFragment(name=name, content=content.strip(), priority=priority)

    def remove_fragment(self, name: str) -> None:
        self.fragments.pop(name, None)

    def build(self) -> str:
        """Returns the assembled system prompt string safely fitting within budget."""
        if not self.fragments:
            return ""
            
        # Sort fragments by priority (descending)
        sorted_fragments = sorted(self.fragments.values(), key=lambda x: x.priority, reverse=True)
        
        selected = []
        current_tokens = 0
        
        for frag in sorted_fragments:
            cost = _estimate_tokens(frag.content) + 5  # +5 for structural newlines
            if current_tokens + cost <= self.max_tokens:
                selected.append(frag)
                current_tokens += cost
            else:
                logger.warning("PromptBuilder: Dropped fragment '%s' to conserve context window.", frag.name)
                
        # We output in order of priority (highest first) since core instructions are usually more important
        parts = [f.content for f in selected]
        return "\n\n".join(parts)
