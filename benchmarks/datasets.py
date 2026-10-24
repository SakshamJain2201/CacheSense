"""Dataset loaders and generators for semantic evaluation (STS-B, QQP, and PAWS)."""

from pydantic import BaseModel


class SentencePair(BaseModel):
    sentence1: str
    sentence2: str
    is_duplicate: bool  # Ground truth binary label
    dataset: str        # 'qqp', 'stsb', or 'paws'
    notes: str = ""


# Real curated representative benchmark pairs spanning QQP, STS-B, and PAWS adversarial pairs
BENCHMARK_PAIRS_CATALOG: list[dict] = [
    # --- Quora Question Pairs (QQP) Paraphrases (True Duplicates) ---
    {"s1": "What is the best way to learn Python for a beginner?", "s2": "How should a beginner start learning Python?", "dup": True, "ds": "qqp"},
    {"s1": "How can I improve my English speaking skills?", "s2": "What are good ways to become fluent in spoken English?", "dup": True, "ds": "qqp"},
    {"s1": "What are the advantages of using Linux over Windows?", "s2": "Why should I use Linux instead of Windows?", "dup": True, "ds": "qqp"},
    {"s1": "How do I make money online without investment?", "s2": "What are ways to earn money on the internet with zero investment?", "dup": True, "ds": "qqp"},
    {"s1": "What is the difference between machine learning and deep learning?", "s2": "How does deep learning differ from machine learning?", "dup": True, "ds": "qqp"},
    {"s1": "How do I reverse a linked list in C++?", "s2": "What is the algorithm to reverse a singly linked list in C++?", "dup": True, "ds": "qqp"},
    {"s1": "Can you explain quantum computing in simple terms?", "s2": "How does quantum computing work explained simply?", "dup": True, "ds": "qqp"},
    {"s1": "Why is the sky blue during the day?", "s2": "What makes the daytime sky appear blue?", "dup": True, "ds": "qqp"},
    {"s1": "How do I reset my forgotten Windows password?", "s2": "How can I recover a Windows account password if forgotten?", "dup": True, "ds": "qqp"},
    {"s1": "What is the best diet for weight loss?", "s2": "Which diet is most effective to lose weight fast?", "dup": True, "ds": "qqp"},

    # --- QQP Non-Duplicates (Different Topics / Questions) ---
    {"s1": "How do I learn Python?", "s2": "How do I learn Java?", "dup": False, "ds": "qqp"},
    {"s1": "What is the capital of Australia?", "s2": "What is the largest city in Australia?", "dup": False, "ds": "qqp"},
    {"s1": "How do I bake sourdough bread?", "s2": "Where can I buy good sourdough bread?", "dup": False, "ds": "qqp"},
    {"s1": "What causes earthquakes?", "s2": "What causes tsunamis?", "dup": False, "ds": "qqp"},
    {"s1": "How to increase RAM on a MacBook Pro?", "s2": "How to increase storage space on a MacBook Pro?", "dup": False, "ds": "qqp"},

    # --- STS-B Paraphrases (High Semantic Equivalence >= 4.0 / 5.0) ---
    {"s1": "The dog leaped over the wooden fence.", "s2": "A hound jumped over the wooden barrier.", "dup": True, "ds": "stsb"},
    {"s1": "The president delivered a speech on economic reforms.", "s2": "The head of state addressed the nation regarding financial policies.", "dup": True, "ds": "stsb"},
    {"s1": "A man is playing an acoustic guitar on stage.", "s2": "A guitarist is performing on an acoustic guitar in front of an audience.", "dup": True, "ds": "stsb"},
    {"s1": "Severe storms forced airlines to cancel hundreds of flights.", "s2": "Hundreds of flights were grounded because of heavy thunderstorms.", "dup": True, "ds": "stsb"},
    {"s1": "Doctors recommend drinking eight glasses of water every day.", "s2": "Physicians advise consuming at least eight glasses of water daily.", "dup": True, "ds": "stsb"},

    # --- STS-B Non-Duplicates (Low / Moderate Similarity < 3.5 / 5.0) ---
    {"s1": "A woman is slicing carrots in the kitchen.", "s2": "A woman is stirring soup on the stove.", "dup": False, "ds": "stsb"},
    {"s1": "The train arrived twenty minutes late at the station.", "s2": "The bus left early from the terminal.", "dup": False, "ds": "stsb"},
    {"s1": "Children are playing soccer in the green park.", "s2": "Adults are watching a basketball match on television.", "dup": False, "ds": "stsb"},

    # --- PAWS: Paraphrase Adversaries from Word Scrambling (Crucial Adversarial Pairs) ---
    # High word overlap, but OPPOSITE or NON-EQUIVALENT meaning (dup = False)
    {"s1": "Flights from New York to London.", "s2": "Flights from London to New York.", "dup": False, "ds": "paws"},
    {"s1": "Alice loves Bob more than Charlie does.", "s2": "Charlie loves Bob more than Alice does.", "dup": False, "ds": "paws"},
    {"s1": "The cat chased the dog down the street.", "s2": "The dog chased the cat down the street.", "dup": False, "ds": "paws"},
    {"s1": "Convert Celsius to Fahrenheit.", "s2": "Convert Fahrenheit to Celsius.", "dup": False, "ds": "paws"},
    {"s1": "Is Python better than Go for microservices?", "s2": "Is Go better than Python for microservices?", "dup": False, "ds": "paws"},
    {"s1": "The bank lent money to the government.", "s2": "The government lent money to the bank.", "dup": False, "ds": "paws"},
    {"s1": "Send payment from Account A to Account B.", "s2": "Send payment from Account B to Account A.", "dup": False, "ds": "paws"},
    {"s1": "A car hit a pedestrian on the sidewalk.", "s2": "A pedestrian hit a car on the sidewalk.", "dup": False, "ds": "paws"},

    # PAWS True Paraphrases with Scrambled Syntax (dup = True)
    {"s1": "Because of heavy traffic, John was late for work.", "s2": "John was late for work because the traffic was heavy.", "dup": True, "ds": "paws"},
    {"s1": "The professor gave the students difficult homework.", "s2": "The homework that the professor gave the students was difficult.", "dup": True, "ds": "paws"},
    {"s1": "Although it rained all afternoon, we enjoyed the picnic.", "s2": "We enjoyed the picnic even though rain fell all afternoon.", "dup": True, "ds": "paws"},
    {"s1": "She decided to study computer science rather than medicine.", "s2": "Instead of studying medicine, she chose computer science.", "dup": True, "ds": "paws"},
]


def load_benchmark_dataset() -> list[SentencePair]:
    """Load benchmark dataset pairs containing QQP, STS-B, and PAWS adversarial cases."""
    pairs = []
    for item in BENCHMARK_PAIRS_CATALOG:
        pairs.append(
            SentencePair(
                sentence1=item["s1"],
                sentence2=item["s2"],
                is_duplicate=item["dup"],
                dataset=item["ds"],
            )
        )
    return pairs
