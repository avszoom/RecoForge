"""Per-category title and body fragments for synthetic item generation.

Templated text — not GPT-generated — so the dataset is deterministic and
free. The fragments are domain-specific enough that a frozen sentence
encoder (MiniLM) will place items from the same category near each other
in embedding space, which is what gives the two-tower model real signal.
"""

from __future__ import annotations

# Per category: a list of (topic, [title_templates], [body_templates]).
# Title templates use {topic} and {modifier}.
# Body templates use {topic}, {modifier}, {entity}.

_MODIFIERS: tuple[str, ...] = (
    "in 2026",
    "explained in plain English",
    "from first principles",
    "for beginners",
    "the production guide",
    "a deep dive",
    "what nobody tells you",
    "lessons learned",
    "without the hype",
    "step by step",
)


CATEGORY_CONTENT: dict[str, dict] = {
    "AI Infrastructure": {
        "topics": (
            "LLM inference cost", "vector databases", "model serving",
            "prompt caching", "RAG pipelines", "GPU scheduling",
            "embedding stores", "fine-tuning workflows",
        ),
        "entities": ("Anthropic", "OpenAI", "Pinecone", "Weaviate", "vLLM", "Triton", "Ray Serve"),
        "title_templates": (
            "Reducing {topic} {modifier}",
            "How we scaled {topic} {modifier}",
            "{topic}: {modifier}",
            "The {topic} problem and what to do about it",
        ),
        "body_templates": (
            "{topic} has become a critical bottleneck. Teams using {entity} are reporting 3x improvements after switching their architecture.",
            "We migrated our {topic} stack to {entity} last quarter. Here is what we learned, and what we would do differently.",
            "If you are building with {entity}, understanding {topic} is non-negotiable. This walkthrough covers the practical tradeoffs.",
        ),
    },

    "Startups": {
        "topics": (
            "fundraising", "product-market fit", "early hiring", "founder dynamics",
            "go-to-market", "B2B sales", "pricing strategy", "seed-to-series-A",
        ),
        "entities": ("YC", "Sequoia", "Stripe", "Vercel", "Notion"),
        "title_templates": (
            "{topic} for first-time founders, {modifier}",
            "What I learned about {topic} {modifier}",
            "Rethinking {topic}: {modifier}",
            "{topic} mistakes nobody warns you about",
        ),
        "body_templates": (
            "After three years working on {topic}, the hardest part was not what we expected. {entity} got it right by focusing on the boring stuff first.",
            "{topic} looks simple from outside. From inside, it is a series of unforced errors. Here is how to avoid the worst of them.",
            "We talked to 40 founders about {topic}. The patterns that emerged surprised us.",
        ),
    },

    "Investing": {
        "topics": (
            "index funds", "asset allocation", "tax-loss harvesting",
            "dividend strategies", "401k optimization", "REIT exposure",
            "rebalancing", "international diversification",
        ),
        "entities": ("Vanguard", "Fidelity", "Schwab", "S&P 500", "Treasury bonds"),
        "title_templates": (
            "{topic}: {modifier}",
            "A practical guide to {topic} {modifier}",
            "Why {topic} matters more than you think",
            "{topic} done right",
        ),
        "body_templates": (
            "{topic} is one of the most misunderstood corners of personal finance. {entity} makes this far easier than it used to be, but the rules of thumb still apply.",
            "If you have not thought about {topic} in the last 12 months, it is worth a look. Allocations drift, fees compound, and {entity} keeps changing the math.",
            "Most retail investors get {topic} wrong in three predictable ways. Here is how to do better.",
        ),
    },

    "Travel": {
        "topics": (
            "long layovers", "shoulder season trips", "budget Europe",
            "solo travel", "family-friendly cities", "rail passes",
            "off-the-grid hikes", "weekend escapes",
        ),
        "entities": ("Lisbon", "Tokyo", "Mexico City", "Edinburgh", "Hanoi", "Patagonia"),
        "title_templates": (
            "{topic} in {entity}",
            "A guide to {topic} {modifier}",
            "Why {entity} is perfect for {topic}",
            "{topic}, rethought",
        ),
        "body_templates": (
            "{entity} surprised us. We had read every guide to {topic}, and still came away with a list of things nobody had mentioned. Worth the trip.",
            "If you have an open weekend, {topic} in {entity} is the move. Cheap flights, walkable streets, and food that ruins you for home.",
            "{topic} is where most travelers cut corners. {entity} is the city to do it right.",
        ),
    },

    "Food": {
        "topics": (
            "knife skills", "weeknight pasta", "sourdough", "kimchi",
            "cast iron care", "stock and broth", "Sichuan cooking", "sheet-pan dinners",
        ),
        "entities": ("Kenji", "Samin", "Bon Appetit", "Serious Eats", "Ottolenghi"),
        "title_templates": (
            "{topic}: {modifier}",
            "Better {topic} starts with one technique",
            "{entity}'s approach to {topic}",
            "The real secret to {topic}",
        ),
        "body_templates": (
            "{topic} is one of those skills that pays back forever. {entity} has been hammering this point for years, and the science backs it up.",
            "Most home cooks get {topic} wrong because the recipes do not explain why. {entity} fixed that, and weeknight dinners have not been the same since.",
            "Forget what you have read about {topic}. The trick is technique, not ingredients. {entity} has the cleanest take.",
        ),
    },

    "Health & Fitness": {
        "topics": (
            "zone 2 cardio", "VO2 max", "strength training basics",
            "sleep quality", "Mediterranean diet", "mobility work",
            "step counts", "creatine for endurance",
        ),
        "entities": ("Peter Attia", "Andrew Huberman", "ACSM", "Stanford Lifestyle"),
        "title_templates": (
            "{topic}: {modifier}",
            "The case for {topic}",
            "How {entity} thinks about {topic}",
            "Rethinking {topic}",
        ),
        "body_templates": (
            "{topic} is suddenly everywhere, and most of what is online is wrong. {entity} has been steady on this for years. Here is the actual evidence.",
            "If you only do one thing for your fitness in the next year, {topic} is a reasonable answer. {entity} explains why better than anyone.",
            "{topic} is not glamorous, but the data is overwhelming. The longer you delay, the harder it gets.",
        ),
    },

    "Programming": {
        "topics": (
            "type systems", "concurrency primitives", "database internals",
            "git rebase workflows", "API design", "observability stacks",
            "profiling tools", "build systems",
        ),
        "entities": ("Rust", "Go", "Postgres", "Linux", "Kubernetes", "Bazel"),
        "title_templates": (
            "{topic} in {entity}, {modifier}",
            "Practical {topic}: {modifier}",
            "{topic} for working engineers",
            "What every senior engineer should know about {topic}",
        ),
        "body_templates": (
            "{topic} is where intuition fails most engineers. {entity} provides a clean enough abstraction that the underlying tradeoffs become visible. Worth your time.",
            "If you are using {entity} in production and have not internalized {topic}, there is a class of bug coming for you eventually.",
            "{topic} sounds boring. Once you have lost a weekend to a {topic} issue, it stops being boring.",
        ),
    },

    "Personal Finance": {
        "topics": (
            "emergency funds", "high-yield savings", "credit card churning",
            "mortgage refinancing", "529 plans", "HSA strategies",
            "debt snowball vs avalanche", "FIRE numbers",
        ),
        "entities": ("Vanguard", "Ally", "Fidelity", "Roth IRA", "I-bonds"),
        "title_templates": (
            "{topic}: {modifier}",
            "A no-nonsense guide to {topic}",
            "The {topic} playbook",
            "Why {topic} is worth a Saturday morning",
        ),
        "body_templates": (
            "{topic} is usually the cheapest, highest-leverage move you can make. {entity} makes the mechanics easy. The hard part is starting.",
            "Most personal-finance content overcomplicates {topic}. The version that actually works fits on an index card.",
            "If your {topic} is not dialed in, every other optimization is premature. {entity} is a fine default.",
        ),
    },

    "Science": {
        "topics": (
            "CRISPR therapeutics", "exoplanet detection", "quantum supremacy claims",
            "fusion timelines", "synthetic biology", "climate models",
            "particle accelerators", "neuroscience of memory",
        ),
        "entities": ("Nature", "MIT", "CERN", "Caltech", "Broad Institute"),
        "title_templates": (
            "{topic}: where we actually are",
            "{topic} {modifier}",
            "What {entity} just published on {topic}",
            "The state of {topic}",
        ),
        "body_templates": (
            "{topic} has been overhyped in the press for a decade. {entity} just published results that change the picture in a way most coverage will get wrong.",
            "{topic} is moving faster than the textbooks. Here is the version of the story that lines up with what {entity} is actually doing.",
            "If you stopped paying attention to {topic} five years ago, the field has changed. {entity} is at the center of the new wave.",
        ),
    },

    "Gaming": {
        "topics": (
            "indie roguelikes", "competitive shooters", "JRPG comebacks",
            "speedrun strategies", "deck builders", "open-world burnout",
            "controller ergonomics", "remasters",
        ),
        "entities": ("Steam Deck", "Nintendo", "FromSoftware", "Larian", "Valve"),
        "title_templates": (
            "{topic} are having a moment",
            "Why {topic} keep working",
            "{entity} and the resurgence of {topic}",
            "{topic}: {modifier}",
        ),
        "body_templates": (
            "{topic} keep eating my weekends, and {entity} is partly to blame. The design pattern works because it respects your time.",
            "If you bounced off {topic} last year, the new wave is different. {entity} has lowered the friction enough that the genre finally clicks.",
            "{topic} are not what they used to be. {entity} is leading a quiet renaissance and most coverage has missed it.",
        ),
    },

    "Music": {
        "topics": (
            "ambient renaissance", "vinyl resurgence", "indie rock revival",
            "live album mixing", "lossless streaming", "guitar pedals",
            "lo-fi production", "concert tour economics",
        ),
        "entities": ("Bandcamp", "Spotify", "Apple Music", "Tidal", "Rough Trade"),
        "title_templates": (
            "{topic} in 2026",
            "Why {topic} are back",
            "{entity} and {topic}",
            "{topic}: {modifier}",
        ),
        "body_templates": (
            "{topic} have been bubbling for a few years and finally feel mainstream again. {entity} has been the quiet driver — independent labels and listener habits both shifting.",
            "{topic} pushed me back into vinyl. {entity} makes it cheap to discover, and the rest takes care of itself.",
            "If you have been on autopilot in your music habits, {topic} is a low-cost way to wake them up. {entity} is the easiest place to start.",
        ),
    },

    "Movies & TV": {
        "topics": (
            "limited series done right", "A24 horror", "legacy sequels",
            "international thrillers", "documentary boom", "rewatchable comedies",
            "streaming wars", "directors to watch",
        ),
        "entities": ("A24", "HBO", "Netflix", "Apple TV+", "Studio Ghibli"),
        "title_templates": (
            "{topic} from {entity}",
            "Why {topic} keep working",
            "{topic}: {modifier}",
            "The case for {topic}",
        ),
        "body_templates": (
            "{topic} have been the surprise of the season. {entity} keeps quietly making them, and the format suits the subject far better than the two-hour film.",
            "If you have not been paying attention to {topic}, {entity} has the strongest current slate. The hit rate is unusually high right now.",
            "{topic} are easy to dismiss. The best of them this year are doing things that movies have stopped trying to do.",
        ),
    },
}


def assert_content_complete(categories: tuple[str, ...]) -> None:
    """Sanity check: content must exist for every taxonomy category."""
    missing = [c for c in categories if c not in CATEGORY_CONTENT]
    if missing:
        raise RuntimeError(f"CATEGORY_CONTENT missing entries for: {missing}")


# Module-level fragments shared across categories.
MODIFIERS: tuple[str, ...] = _MODIFIERS
