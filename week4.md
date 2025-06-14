# Week 4: Privacy-preserving Data Publishing III

<div align="center">

[⬅️ **Week 3**](week3.md) | [**Main**](README.md) | [**Week 5** ➡️](week5.md)

</div>

---

## 🎯 Learning Goals

By the end of this week, you should understand:
- Differential privacy: fundamental concepts and mathematical foundations
- Privacy budget and composition theorems
- Mechanisms for achieving differential privacy (Laplace, Gaussian)
- Practical applications and limitations of differential privacy

---

## 📖 Theoretical Content

### Introduction to Differential Privacy

**Motivation:** Traditional anonymization methods (k-anonymity, l-diversity, t-closeness) are vulnerable to auxiliary information attacks. Differential privacy provides **mathematical guarantees** regardless of what background knowledge an attacker might have.

**Core Philosophy:** The presence or absence of any individual's data should not significantly affect the output of any analysis.

**Formal Definition (ε-Differential Privacy):**
A randomized algorithm M satisfies ε-differential privacy if for all datasets D₁ and D₂ that differ by at most one record, and for all possible outputs S:

**Pr[M(D₁) ∈ S] ≤ exp(ε) × Pr[M(D₂) ∈ S]**

### Key Concepts

**1. Privacy Parameter (ε)**
- **Small ε:** Strong privacy protection (outputs are nearly identical)
- **Large ε:** Weaker privacy but higher utility
- **Common values:** ε ∈ [0.01, 10]
- **Rule of thumb:** ε ≤ 1 provides meaningful privacy

**2. Adjacent Datasets**
- Two datasets that differ in exactly one individual's record
- Captures the worst-case privacy loss from participation
- Includes both addition and removal of records

**3. Randomized Mechanism**
- Must add carefully calibrated noise
- Noise amount depends on ε and query sensitivity
- Larger ε allows less noise (better utility)

### Global Sensitivity

**Definition:** The maximum change in query output when one record is added or removed from any dataset.

**For Count Queries:** GS = 1
- Adding/removing one person changes count by at most 1

**For Sum Queries:** GS = max possible individual contribution
- If salary range is [0, 200K], then GS = 200,000

**For Average Queries:** More complex
- Depends on dataset size and value range
- Often requires bounding techniques

### The Laplace Mechanism

**For Numeric Queries:**
Add noise drawn from Laplace distribution with scale b = GS/ε

**Laplace Distribution:**
- **PDF:** f(x) = (1/2b) × exp(-|x|/b)
- **Mean:** 0 (noise is unbiased)
- **Variance:** 2b²

**Algorithm:**
1. Compute true query result f(D)
2. Generate noise η ~ Laplace(GS/ε)
3. Return f(D) + η

**Example - Count Query:**
- True count: 1,247 people
- ε = 1, GS = 1
- Noise scale: b = 1/1 = 1
- Sample noise: η = 2.3
- Output: 1,247 + 2.3 = 1,249.3

### The Gaussian Mechanism

**For (ε, δ)-Differential Privacy:**
A relaxed version where privacy can fail with probability δ

**Noise Scale:** σ = √(2 ln(1.25/δ)) × GS/ε

**When to Use:**
- When pure ε-DP is too restrictive
- For complex queries requiring less noise
- δ is typically very small (e.g., 10⁻⁵)

### Privacy Budget and Composition

**Sequential Composition:**
If you run k algorithms with privacy parameters ε₁, ε₂, ..., εₖ, the total privacy cost is:
**ε_total = ε₁ + ε₂ + ... + εₖ**

**Privacy Budget Management:**
- Start with total budget (e.g., ε = 1)
- Allocate portions to different queries
- Once budget is exhausted, no more queries allowed
- Critical for long-term data usage

**Advanced Composition:**
- Parallel composition: Independent datasets can be queried with same ε
- Group privacy: Privacy degrades for correlated individuals

---

## 🔍 Detailed Explanations

### Understanding the ε Parameter

**Intuitive Interpretation:**
- **ε = 0:** Perfect privacy (but useless utility)
- **ε = 0.1:** Very strong privacy, significant noise
- **ε = 1:** Reasonable privacy-utility balance
- **ε = 10:** Weak privacy, minimal noise

**Mathematical Meaning:**
exp(ε) represents the maximum ratio between probabilities of any outcome for adjacent datasets.

**Example with ε = 1:**
- exp(1) ≈ 2.72
- Any outcome can be at most 2.72× more likely with/without your data
- Provides plausible deniability

### Calibrating Noise for Different Queries

**Count Query Example:**
```python
def dp_count(data, predicate, epsilon):
    true_count = sum(1 for record in data if predicate(record))
    noise = np.random.laplace(0, 1/epsilon)  # GS = 1
    return true_count + noise
```

**Sum Query Example:**
```python
def dp_sum(data, attribute, max_value, epsilon):
    true_sum = sum(record[attribute] for record in data)
    noise = np.random.laplace(0, max_value/epsilon)  # GS = max_value
    return true_sum + noise
```

**Average Query Example:**
```python
def dp_average(data, attribute, max_value, epsilon):
    n = len(data)
    dp_sum_result = dp_sum(data, attribute, max_value, epsilon/2)
    dp_count_result = dp_count(data, lambda x: True, epsilon/2)
    return dp_sum_result / dp_count_result
```

### Privacy Budget Allocation Strategies

**Uniform Allocation:**
- Divide budget equally among all planned queries
- Simple but may not be optimal

**Adaptive Allocation:**
- Allocate more budget to more important queries
- Requires knowing query priorities in advance

**Hierarchical Allocation:**
- Use tree structure for range queries
- Efficient for many related queries

---

## 💡 Practical Examples

### Example 1: Census Data Release

**Scenario:** Releasing population statistics while protecting individual privacy

**Traditional Approach:**
```
Query: "How many people in ZIP 02139 earn >$100K?"
Answer: 847 people
```

**Differential Privacy Approach (ε = 1):**
```python
true_count = 847
sensitivity = 1  # Adding/removing one person changes count by 1
epsilon = 1.0
noise = laplace_noise(scale=sensitivity/epsilon)  # scale = 1
noisy_count = 847 + noise

# Possible outputs: 846.3, 849.7, 845.1, etc.
```

**Privacy Guarantee:** Whether any specific individual is included in the dataset, the probability of any particular answer changes by at most factor of e ≈ 2.72.

### Example 2: Medical Research Query

**Scenario:** Researcher wants to know average age of patients with diabetes

**Setup:**
- Dataset: 10,000 patient records
- Age range: [0, 120] years
- ε = 0.5 (strong privacy)

**Implementation:**
```python
# Step 1: Count diabetic patients
diabetic_count = dp_count(patients, 
                         lambda p: p.diagnosis == "diabetes", 
                         epsilon=0.25)

# Step 2: Sum ages of diabetic patients  
diabetic_age_sum = dp_sum(patients.filter(diabetes), 
                         "age", 
                         max_value=120, 
                         epsilon=0.25)

# Step 3: Compute average
average_age = diabetic_age_sum / diabetic_count
```

### Example 3: Location Analytics

**Scenario:** City planning department analyzing foot traffic

**Query:** "How many people visited downtown area between 2-4 PM?"

**Challenges:**
- Location data is highly sensitive
- Need temporal analysis
- Multiple related queries planned

**Solution:**
```python
epsilon_total = 1.0
time_slots = 12  # 2-hour periods in a day
epsilon_per_slot = epsilon_total / time_slots

for slot in time_slots:
    traffic_count = dp_count(location_data, 
                           lambda record: is_downtown(record, slot),
                           epsilon_per_slot)
    publish_result(slot, traffic_count)
```

---

## ❓ Self-Assessment Questions

<details>
<summary><b>Question 1:</b> What is the fundamental difference between differential privacy and traditional anonymization methods like k-anonymity? (Click to reveal answer)</summary>

**Answer:** 
**Traditional Methods (k-anonymity, l-diversity, t-closeness):**
- Syntactic privacy: Modify data structure to prevent re-identification
- Vulnerable to auxiliary information attacks
- No mathematical guarantees about privacy protection
- Protection depends on assumptions about attacker knowledge

**Differential Privacy:**
- Semantic privacy: Provides mathematical guarantees regardless of auxiliary information
- Privacy protection doesn't depend on what the attacker knows
- Quantifiable privacy loss through ε parameter
- Robust against any background knowledge attack
- Trade-off: Requires adding random noise, reducing utility

**Key Insight:** Differential privacy protects against the **worst-case** attacker with unlimited background knowledge, while traditional methods protect against specific attack models.

</details>

<details>
<summary><b>Question 2:</b> Calculate the noise scale for a sum query where individual contributions are bounded by 50,000 and ε = 0.5. What's the standard deviation of the added noise? (Click to reveal answer)</summary>

**Answer:** 
**Given:**
- Global Sensitivity (GS) = 50,000 (max individual contribution)
- ε = 0.5

**Laplace Mechanism:**
- Noise scale: b = GS/ε = 50,000/0.5 = 100,000
- Standard deviation: σ = √2 × b = √2 × 100,000 ≈ 141,421

**Interpretation:** The added noise has a standard deviation of about 141,421, which is quite large compared to typical individual contributions. This illustrates the trade-off between privacy (small ε) and utility - stronger privacy requires more noise.

For context, about 68% of noise values will be within ±141,421 of zero, and 95% within ±282,842.

</details>

<details>
<summary><b>Question 3:</b> You have a privacy budget of ε = 1 and need to answer 10 queries. How would you allocate the budget, and what are the implications? (Click to reveal answer)</summary>

**Answer:** 
**Uniform Allocation:**
- Each query gets ε/10 = 0.1
- Very strong privacy per query but significant noise
- Total privacy cost: 10 × 0.1 = 1.0

**Non-Uniform Allocation Examples:**
1. **Priority-based:** 
   - Important queries: ε = 0.3 (3 queries)
   - Regular queries: ε = 0.1 (1 query)
   - Total: 3×0.3 + 1×0.1 = 1.0

2. **Hierarchical:**
   - Use tree-based mechanisms for related queries
   - Can answer more queries with same total budget

**Implications:**
- **High noise:** Each query has substantial noise with ε = 0.1
- **Limited utility:** Individual query results may be unreliable
- **No additional queries:** Budget exhausted after planned queries
- **Strategic planning:** Need to prioritize most important analyses

**Alternative:** Consider using (ε,δ)-differential privacy with δ = 10⁻⁵ to reduce noise while maintaining strong privacy.

</details>

<details>
<summary><b>Question 4:</b> Explain why differential privacy is "composable" and why this matters for practical deployments. (Click to reveal answer)</summary>

**Answer:** 
**Composability** means that when multiple differentially private algorithms are applied to the same dataset, the total privacy loss can be calculated and bounded.

**Sequential Composition:**
- Run k algorithms with privacy parameters ε₁, ε₂, ..., εₖ
- Total privacy cost: ε_total = ε₁ + ε₂ + ... + εₖ
- Privacy degrades additively

**Why This Matters:**

1. **Long-term Privacy Accounting:** Organizations can track cumulative privacy loss over time
2. **Budget Management:** Can allocate privacy budget across different users/queries
3. **Formal Guarantees:** Mathematical proof that privacy doesn't degrade faster than linear rate
4. **System Design:** Enables building complex privacy-preserving systems with known guarantees

**Example:** A hospital database used for research:
- Year 1: ε₁ = 0.3 for cancer study
- Year 2: ε₂ = 0.4 for diabetes study  
- Year 3: ε₃ = 0.3 for heart disease study
- Total privacy loss: ε_total = 1.0

After this, the dataset has "exhausted" its privacy budget and no more queries should be allowed without additional privacy protections.

</details>

<details>
<summary><b>Question 5:</b> Compare the privacy-utility trade-offs of ε = 0.1, ε = 1, and ε = 10 for a count query with 1000 true results. (Click to reveal answer)</summary>

**Answer:** 
**For Count Query (GS = 1):**

**ε = 0.1 (Very Strong Privacy):**
- Noise scale: b = 1/0.1 = 10
- Standard deviation: ≈ 14.14
- Typical results: 1000 ± 28 (95% confidence)
- **Privacy:** Excellent - outcomes nearly identical with/without any individual
- **Utility:** Poor - high relative noise (±2.8%)

**ε = 1 (Balanced):**
- Noise scale: b = 1/1 = 1  
- Standard deviation: ≈ 1.41
- Typical results: 1000 ± 3 (95% confidence)
- **Privacy:** Good - max probability ratio of e ≈ 2.72
- **Utility:** Good - low relative noise (±0.3%)

**ε = 10 (Weak Privacy):**
- Noise scale: b = 1/10 = 0.1
- Standard deviation: ≈ 0.14
- Typical results: 1000 ± 0.3 (95% confidence)
- **Privacy:** Weak - max probability ratio of e¹⁰ ≈ 22,026
- **Utility:** Excellent - minimal noise (±0.03%)

**Recommendation:** ε = 1 offers the best balance for most applications.

</details>

---

## 🔬 Lab Exercises

### Exercise 1: Implementing Basic Differential Privacy

**Task:** Implement the Laplace mechanism for different query types

```python
import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    Add Laplace noise to achieve epsilon-differential privacy
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise

# Test with different parameters
true_count = 1000
for eps in [0.1, 1.0, 10.0]:
    noisy_result = laplace_mechanism(true_count, 1, eps)
    print(f"ε={eps}: {true_count} → {noisy_result:.2f}")
```

**Your Tasks:**
1. Run the code with different ε values
2. Observe the noise magnitude changes
3. Calculate theoretical vs empirical variance
4. Implement sum and average queries

### Exercise 2: Privacy Budget Management

**Scenario:** You have ε = 2.0 budget for analyzing a customer dataset

**Planned Queries:**
1. Total number of customers
2. Average age of customers
3. Number of customers in each city (10 cities)
4. Average spending by age group (5 groups)

**Your Task:** Design a budget allocation strategy and justify your choices.

### Exercise 3: Composition Analysis

Calculate the total privacy cost for this sequence:
1. 5 count queries, each with ε = 0.2
2. 3 sum queries, each with ε = 0.3
3. 1 average query with ε = 0.1

What's the remaining budget if you started with ε = 2.0?

---

## 📚 Additional Resources

### Foundational Papers
- Dwork, C. (2006). "Differential Privacy"
- Dwork, C. et al. (2014). "The Algorithmic Foundations of Differential Privacy"

### Practical Guides
- Wood, A. et al. (2018). "Differential Privacy: A Primer for a Non-Technical Audience"
- Hsu, J. et al. (2014). "Differential Privacy: An Economic Method for Choosing Epsilon"

### Tools and Libraries
- **Google's Differential Privacy Library:** Open-source implementations
- **OpenDP:** Modular differential privacy library
- **IBM's Diffprivlib:** Python library for differential privacy
- **Microsoft's SmartNoise:** Differential privacy platform

### Real-World Applications
- **U.S. Census 2020:** First major use of differential privacy for official statistics
- **Apple's Local Differential Privacy:** Used in iOS for telemetry
- **Google's RAPPOR:** Privacy-preserving analytics

---

## 📋 Case Study Discussion Preparation

**Case Study 1: Netflix Prize Dataset Revisited**

**Background:** How would differential privacy have prevented the Narayanan-Shmatikov attack?

**Discussion Points:**
1. **Traditional Anonymization Failures:**
   - Removed direct identifiers only
   - Vulnerable to auxiliary information (IMDb data)
   - No mathematical privacy guarantees

2. **Differential Privacy Solution:**
   - Add noise to recommendation algorithms
   - Limit number of queries per user
   - Provide formal privacy guarantees

3. **Implementation Challenges:**
   - Balancing recommendation quality with privacy
   - Managing privacy budget across time
   - Handling sparse rating matrices

**Questions for Discussion:**
- What ε value would you recommend for Netflix data?
- How would differential privacy affect recommendation accuracy?
- What are the business implications of formal privacy guarantees?

**Preparation:** Review the original attack paper and consider how different privacy mechanisms would have prevented it.

---

## 🚀 Project 1 Final Phase

**Final Week Checklist:**
- [ ] Complete differential privacy implementation
- [ ] Compare k-anonymity, l-diversity, t-closeness, and differential privacy
- [ ] Analyze privacy-utility trade-offs across all methods
- [ ] Prepare comprehensive evaluation metrics
- [ ] Finalize report with conclusions and recommendations
- [ ] Practice presentation for next week's defenses

**Report Structure:**
1. **Introduction:** Dataset description and anonymization goals
2. **Methods:** Implementation of all four approaches
3. **Evaluation:** Privacy guarantees and utility metrics
4. **Comparison:** Trade-offs between different methods
5. **Conclusion:** Recommendations for practical deployment

**Evaluation Metrics:**
- **Privacy:** Re-identification risk, inferential disclosure risk
- **Utility:** Query accuracy, statistical properties preservation
- **Efficiency:** Computational cost, scalability

---

<div align="center">

[⬅️ **Week 3**](week3.md) | [**Main**](README.md) | [**Week 5** ➡️](week5.md)

</div>