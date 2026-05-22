#!/usr/bin/env python3
"""
ADVANCED Black-box Attack - Multiple Strategies
Uses adaptive word selection and strategic placement
"""

import os
import requests
import random
from collections import Counter

BASE_URL = os.getenv("BASE_URL", "http://154.57.164.71:31550")

def predict(text: str) -> dict:
    r = requests.post(f"{BASE_URL}/predict", json={"text": text})
    return r.json()

def get_challenge(phase: str) -> dict:
    r = requests.get(f"{BASE_URL}/challenge/{phase}")
    return r.json()

def submit_solutions(phase: str, solutions: list) -> dict:
    r = requests.post(f"{BASE_URL}/submit/{phase}", json={"solutions": solutions})
    return r.json()

# ============================================================================
# EXPANDED POSITIVE VOCABULARY - More diverse and powerful
# ============================================================================

TIER_1_WORDS = [
    # Strongest universal positive words
    "excellent", "outstanding", "superb", "magnificent", "exceptional",
    "brilliant", "wonderful", "fantastic", "amazing", "incredible",
    "phenomenal", "spectacular", "marvelous", "splendid", "fabulous",
]

TIER_2_WORDS = [
    # Strong positive but slightly less intense
    "great", "awesome", "terrific", "impressive", "remarkable",
    "delightful", "lovely", "beautiful", "perfect", "stunning",
    "gorgeous", "best", "top", "favorite", "recommended",
]

TIER_3_WORDS = [
    # Positive emotions and reactions
    "love", "loved", "enjoy", "enjoyed", "adore", "adored",
    "appreciate", "appreciate", "thank", "thankful", "grateful",
    "pleased", "happy", "glad", "satisfied", "delighted",
]

TIER_4_WORDS = [
    # Positive adjectives
    "good", "nice", "fine", "pleasant", "enjoyable",
    "entertaining", "engaging", "captivating", "compelling", "interesting",
    "fun", "exciting", "thrilling", "inspiring", "uplifting",
]

# Combine all tiers
ALL_POSITIVE_WORDS = TIER_1_WORDS + TIER_2_WORDS + TIER_3_WORDS + TIER_4_WORDS


# ============================================================================
# SMART WORD TESTING AND RANKING
# ============================================================================

def find_most_effective_words(sample_text: str, candidate_words: list, top_k: int = 50):
    """
    Test words on sample text to find most effective ones
    """
    print(f"\n[*] Testing {len(candidate_words)} candidate words on sample...")
    
    baseline = predict(sample_text)
    baseline_prob = baseline.get('positive_probability', 0.0)
    print(f"    Baseline: {baseline_prob:.4f}")
    
    word_scores = []
    for i, word in enumerate(candidate_words):
        test = predict(f"{sample_text} {word}")
        impact = test.get('positive_probability', 0.0) - baseline_prob
        word_scores.append((word, impact))
        
        if (i + 1) % 20 == 0:
            print(f"    Tested {i+1}/{len(candidate_words)} words...")
    
    # Sort by impact
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n[*] Top 15 most effective words:")
    for word, impact in word_scores[:15]:
        print(f"    '{word}': {impact:+.4f}")
    
    return [w for w, _ in word_scores[:top_k]]


# ============================================================================
# ATTACK STRATEGIES
# ============================================================================

def strategy_1_simple_append(text: str, words: list, budget: int) -> str:
    """Strategy 1: Simple append at the end"""
    return f"{text} {' '.join(words[:budget])}"


def strategy_2_repeated_words(text: str, words: list, budget: int) -> str:
    """Strategy 2: Repeat most effective words"""
    # Take top 5 words and repeat them
    top_words = words[:5]
    repeated = []
    while len(repeated) < budget:
        repeated.extend(top_words)
    return f"{text} {' '.join(repeated[:budget])}"


def strategy_3_sentence_style(text: str, words: list, budget: int) -> str:
    """Strategy 3: Make it look like sentences"""
    selected = words[:budget]
    # Group into sentence-like chunks
    chunks = []
    for i in range(0, len(selected), 5):
        chunk = ' '.join(selected[i:i+5])
        chunks.append(f"{chunk}.")
    
    return f"{text} {' '.join(chunks)}"


def strategy_4_editor_note(text: str, words: list, budget: int) -> str:
    """Strategy 4: Add as editor's note"""
    selected = words[:budget]
    return f"{text} [Editor: {' '.join(selected)}]"


def strategy_5_mixed_placement(text: str, words: list, budget: int) -> str:
    """Strategy 5: Mix words throughout"""
    selected = words[:budget]
    # Split text into sentences
    parts = text.split('. ')
    
    if len(parts) > 1:
        # Insert words between sentences
        result = parts[0]
        word_idx = 0
        for part in parts[1:]:
            if word_idx < len(selected):
                result += f". {selected[word_idx]} {part}"
                word_idx += 1
            else:
                result += f". {part}"
        
        # Add remaining words at end
        if word_idx < len(selected):
            result += " " + " ".join(selected[word_idx:])
        
        return result
    else:
        return strategy_1_simple_append(text, words, budget)


def strategy_6_power_words(text: str, words: list, budget: int) -> str:
    """Strategy 6: Use only the most powerful words, repeated"""
    # Just use tier 1 words, repeated many times
    power = words[:10]
    repeated = (power * (budget // len(power) + 1))[:budget]
    return f"{text} {' '.join(repeated)}"


# ============================================================================
# ADAPTIVE ATTACK - Try multiple strategies
# ============================================================================

def adaptive_attack(original_text: str, effective_words: list, budget: int):
    """
    Try multiple strategies and pick the best one
    """
    print(f"\n[*] Trying multiple attack strategies...")
    
    strategies = [
        ("Simple Append", strategy_1_simple_append),
        ("Repeated Words", strategy_2_repeated_words),
        ("Sentence Style", strategy_3_sentence_style),
        ("Editor Note", strategy_4_editor_note),
        ("Mixed Placement", strategy_5_mixed_placement),
        ("Power Words", strategy_6_power_words),
    ]
    
    best_text = None
    best_prob = 0.0
    best_strategy = None
    
    for name, strategy_func in strategies:
        augmented = strategy_func(original_text, effective_words, budget)
        result = predict(augmented)
        prob = result.get('positive_probability', 0.0)
        label = result['label']
        
        print(f"    {name:20} -> {label:8} (prob={prob:.4f})")
        
        if prob > best_prob:
            best_prob = prob
            best_text = augmented
            best_strategy = name
    
    print(f"\n[✓] Best strategy: {best_strategy} (prob={best_prob:.4f})")
    return best_text, best_prob


# ============================================================================
# ULTIMATE FALLBACK - If nothing works, try everything
# ============================================================================

def nuclear_option(text: str, budget: int) -> str:
    """
    Nuclear option: use ALL the most powerful words we have
    """
    print(f"\n[!] NUCLEAR OPTION - Using strongest possible attack")
    
    # Use only tier 1 words, repeated to fill budget
    power_words = TIER_1_WORDS * (budget // len(TIER_1_WORDS) + 1)
    selected = power_words[:budget]
    
    # Try multiple combinations
    options = [
        f"{text} {' '.join(selected)}",
        f"{text} " + " ".join([f"{w}!" for w in selected[:20]]),  # Add emphasis
        f"{text} Correction: {' '.join(selected)}",
        f"{text} Update: {' '.join(selected)}",
    ]
    
    best = options[0]
    best_prob = 0.0
    
    for opt in options:
        result = predict(opt)
        prob = result.get('positive_probability', 0.0)
        if prob > best_prob:
            best_prob = prob
            best = opt
    
    print(f"[✓] Nuclear option best prob: {best_prob:.4f}")
    return best


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ADVANCED BLACK-BOX ATTACK - Multi-Strategy Approach")
    print("="*80)
    
    # Load challenge
    challenge = get_challenge('blackbox')
    reviews = challenge['reviews']
    budget = challenge['max_added_words']
    
    print(f"\n[*] Challenge loaded:")
    print(f"    Reviews: {len(reviews)}")
    print(f"    Budget: {budget} words per review")
    
    # Find most effective words using first review as sample
    sample_review = reviews[0]['text']
    print(f"\n[*] Finding most effective words using sample review...")
    effective_words = find_most_effective_words(sample_review, ALL_POSITIVE_WORDS, top_k=60)
    
    # Attack each review
    solutions = []
    success_count = 0
    
    for i, review in enumerate(reviews):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(reviews)}] Attacking: {review['id']}")
        print(f"Text preview: {review['text'][:100]}...")
        
        # Try adaptive attack
        augmented, prob = adaptive_attack(review['text'], effective_words, budget)
        
        # Check if successful
        result = predict(augmented)
        if result['label'] == 'positive':
            print(f"[✓] SUCCESS: Flipped to positive!")
            success_count += 1
        else:
            print(f"[!] FAILED: Still {result['label']} (prob={prob:.4f})")
            print(f"[!] Trying nuclear option...")
            augmented = nuclear_option(review['text'], budget)
            result = predict(augmented)
            print(f"[*] Nuclear result: {result['label']} (prob={result.get('positive_probability', 0.0):.4f})")
            
            if result['label'] == 'positive':
                success_count += 1
        
        solutions.append({
            "id": review['id'],
            "augmented_text": augmented
        })
    
    # Submit
    print(f"\n{'='*80}")
    print(f"[*] Submission Summary:")
    print(f"    Successfully flipped: {success_count}/{len(reviews)}")
    print(f"[*] Submitting solutions...")
    
    result = submit_solutions('blackbox', solutions)
    print(f"\n[✓] Result: {result}")


if __name__ == "__main__":
    main()
