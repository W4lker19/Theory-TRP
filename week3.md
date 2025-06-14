# Week 3: Privacy-preserving Data Publishing II

<div align="center">
  <a href="week2.md">⬅️ <strong>Week 2</strong></a> |
  <a href="README.md"><strong>Main</strong></a> |
  <a href="week4.md"><strong>Week 4</strong> ➡️</a>
</div>

---

## 🎯 Learning Goals

By the end of this week, you should understand:
- The limitations of k-anonymity and why it's insufficient
- l-diversity: definition, types, and implementation
- t-closeness: concept and motivation
- Advanced anonymization techniques and their trade-offs

---

## 📖 Theoretical Content

### Limitations of k-Anonymity

While k-anonymity prevents record linkage, it has several critical vulnerabilities:

**1. Homogeneity Attack (Lack of Diversity)**
- All records in an equivalence class have the same sensitive value
- Attacker can infer sensitive information without re-identification
- Example: All patients in a group have the same disease

**2. Background Knowledge Attack**
- Attacker has additional information about individuals
- Can eliminate possibilities and narrow down sensitive values
- External knowledge reduces the protection of k-anonymity

**3. Skewness Attack**
- Sensitive attribute distribution is not uniform
- Some values are more common than others
- Probabilistic inference becomes possible

### l-Diversity Model

**Definition:** An equivalence class satisfies l-diversity if it contains at least l "well-represented" values for each sensitive attribute.

**Types of l-Diversity:**

**1. Distinct l-Diversity**
- Simplest form: at least l distinct sensitive values
- Each equivalence class has ≥ l different sensitive attribute values
- Easy to implement but may not prevent probabilistic inference

**2. Entropy l-Diversity**
- Uses entropy to measure diversity
- Entropy(equivalence class) ≥ log(l)
- Better protection against probabilistic attacks
- Formula: H = -Σ(pi × log(pi)) where pi is probability of value i

**3. Recursive (c,l)-Diversity**
- Most frequent value appears ≤ c times more than least frequent
- Provides stronger protection against skewness
- More complex but offers better utility-privacy balance

### t-Closeness Model

**Definition:** An equivalence class satisfies t-closeness if the distance between the distribution of sensitive attributes in the class and the distribution in the entire dataset is ≤ t.

**Key Concepts:**
- **Earth Mover's Distance (EMD):** Measures distribution similarity
- **Global Distribution:** Sensitive attribute distribution in full dataset
- **Local Distribution:** Distribution within equivalence class

**Advantages:**
- Prevents attribute disclosure through distribution analysis
- Handles both categorical and numerical sensitive attributes
- Provides stronger privacy guarantees than l-diversity

**Challenges:**
- More restrictive than l-diversity
- Can significantly reduce data utility
- Complex parameter selection (choosing appropriate t)

### Advanced Anonymization Techniques

**1. Anatomy**
- Separates quasi-identifiers from sensitive attributes
- Creates two tables: QID table and sensitive table
- Uses group identifiers to link tables
- Provides flexibility in sensitive attribute handling

**2. Permutation**
- Randomly permutes sensitive attribute values within groups
- Maintains statistical properties
- Breaks individual-level associations
- Useful for certain types of analysis

**3. (α,k)-Anonymity**
- Combines k-anonymity with confidence bounds
- Limits confidence of inferring sensitive values to α
- More flexible than strict l-diversity requirements

---

## 🔍 Detailed Explanations

### Understanding Homogeneity Attacks

**Scenario:** Medical dataset with k=3 anonymity

**Vulnerable Equivalence Class:**
```
Age Group | Gender | ZIP Area | Disease
20-30     | Male   | 1300*    | HIV
20-30     | Male   | 1300*    | HIV  
20-30     | Male   | 1300*    | HIV
```

**Attack:** Even without knowing which specific record belongs to the target, an attacker knows that any male aged 20-30 from ZIP 1300* has HIV.

**l-Diversity Solution (l=2):**
```
Age Group | Gender | ZIP Area | Disease
20-30     | Male   | 1300*    | HIV
20-30     | Male   | 1300*    | Diabetes
20-30     | Male   | 1300*    | Flu
```

Now each group has at least 2 distinct diseases, preventing certain inference.

### Background Knowledge Attack Example

**Published k-Anonymous Data:**
```
Age Group | Gender | ZIP | Disease
20-25     | Female | 130** | Heart Disease
20-25     | Female | 130** | Diabetes
20-25     | Female | 130** | Flu
```

**Attacker's Background Knowledge:**
- Target: Alice, 23-year-old female from ZIP 13001
- Additional info: Alice doesn't have diabetes (from conversation)

**Attack:** Attacker knows Alice is in this group and eliminates diabetes, leaving 50% chance of heart disease vs flu - significant privacy loss.

### Entropy l-Diversity Calculation

**Example Equivalence Class:**
- Disease A: 2 patients
- Disease B: 1 patient  
- Disease C: 1 patient
- Total: 4 patients

**Entropy Calculation:**
- P(A) = 2/4 = 0.5
- P(B) = 1/4 = 0.25
- P(C) = 1/4 = 0.25

**Entropy = -(0.5×log₂(0.5) + 0.25×log₂(0.25) + 0.25×log₂(0.25))**
**= -(0.5×(-1) + 0.25×(-2) + 0.25×(-2))**
**= -(-0.5 - 0.5 - 0.5) = 1.5**

For l=2: log₂(2) = 1
Since 1.5 > 1, this satisfies 2-diversity.

---

## 💡 Practical Examples

### Example 1: Implementing l-Diversity

**Original Data:**
```
Patient | Age | Gender | ZIP   | Salary | Disease
A       | 23  | M      | 02139 | 45K    | Asthma
B       | 24  | M      | 02139 | 47K    | Asthma
C       | 25  | F      | 02142 | 55K    | Diabetes
D       | 26  | F      | 02142 | 57K    | Heart Disease
```

**After 2-Anonymity (from Week 2):**
```
Group | Age   | Gender | ZIP   | Salary | Disease
1     | 20-25 | M      | 0213* | 46K    | Asthma
1     | 20-25 | M      | 0213* | 46K    | Asthma
2     | 25-30 | F      | 0214* | 56K    | Diabetes
2     | 25-30 | F      | 0214* | 56K    | Heart Disease
```

**Problem:** Group 1 has homogeneity attack vulnerability (both have Asthma)

**2-Diversity Solution:** Need to reorganize groups
```
Group | Age   | Gender | ZIP   | Salary   | Disease
1     | 20-30 | *      | 021** | 45-55K   | Asthma
1     | 20-30 | *      | 021** | 45-55K   | Diabetes
2     | 20-30 | *      | 021** | 47-57K   | Asthma
2     | 20-30 | *      | 021** | 47-57K   | Heart Disease
```

### Example 2: t-Closeness Analysis

**Global Disease Distribution:**
- Flu: 40%
- Diabetes: 30%
- Heart Disease: 20%
- Cancer: 10%

**Equivalence Class Distribution:**
- Flu: 50%
- Diabetes: 25%
- Heart Disease: 25%
- Cancer: 0%

**Earth Mover's Distance Calculation:**
- Move 10% from Flu to Cancer: cost = 10% × 3 = 0.3
- Move 5% from Diabetes to Cancer: cost = 5% × 2 = 0.1
- Total EMD = 0.4

If t = 0.3, this class violates t-closeness (0.4 > 0.3)

### Example 3: Utility Impact Comparison

**Dataset:** 10,000 patient records

**Anonymization Results:**
```
Method          | Groups | Avg Group Size | Info Loss | Privacy Level
k=5             | 500    | 20            | Low       | Basic
(5,2)-diversity | 800    | 12.5          | Medium    | Better
0.2-closeness   | 1200   | 8.3           | High      | Strongest
```

**Analysis:**
- Higher privacy protection requires more groups
- Smaller groups mean more generalization
- Trade-off between utility and privacy protection

---

## ❓ Self-Assessment Questions

<details>
<summary><b>Question 1:</b> Explain the homogeneity attack and how l-diversity addresses it. (Click to reveal answer)</summary>

**Answer:** 
**Homogeneity Attack:** Occurs when all records in a k-anonymous equivalence class have the same sensitive attribute value. Even though individual records cannot be re-identified, an attacker can infer the sensitive value for anyone in that group.

**Example:** If all records in a group have "Cancer" as the disease, knowing someone is in that group reveals their disease.

**l-Diversity Solution:** Requires each equivalence class to have at least l distinct values for sensitive attributes. This ensures that even if someone is known to be in a particular group, there are multiple possible sensitive values, preventing certain inference. For l=3, each group must have at least 3 different diseases, so no single disease can be inferred with certainty.

</details>

<details>
<summary><b>Question 2:</b> What is the difference between distinct l-diversity and entropy l-diversity? (Click to reveal answer)</summary>

**Answer:** 
**Distinct l-Diversity:**
- Simply requires at least l different sensitive values in each equivalence class
- Doesn't consider the distribution of these values
- Vulnerable to skewed distributions (e.g., 99 records with Disease A, 1 with Disease B satisfies 2-diversity but offers little protection)

**Entropy l-Diversity:**
- Uses information entropy to measure diversity: Entropy ≥ log(l)
- Considers both the number of distinct values AND their distribution
- Provides better protection against probabilistic inference
- More robust against skewness attacks
- Example: A group with uniform distribution of l values has maximum entropy = log(l)

Entropy l-diversity is stronger because it prevents attackers from making confident probabilistic inferences based on value frequencies.

</details>

<details>
<summary><b>Question 3:</b> A dataset has the global distribution: Disease A (60%), Disease B (30%), Disease C (10%). An equivalence class has: Disease A (80%), Disease B (20%), Disease C (0%). Calculate the Earth Mover's Distance. (Click to reveal answer)</summary>

**Answer:** 
**Earth Mover's Distance (EMD) Calculation:**

We need to transform the class distribution to match the global distribution:

**Current:** A(80%), B(20%), C(0%)  
**Target:** A(60%), B(30%), C(10%)

**Transformations needed:**
1. Move 20% from A to B: A→B = 20% × distance(A,B) = 20% × 1 = 0.20
2. Move 10% from A to C: A→C = 10% × distance(A,C) = 10% × 2 = 0.20

**Total EMD = 0.20 + 0.20 = 0.40**

Note: Distances between diseases are typically: adjacent=1, non-adjacent=2 (or based on semantic similarity). The EMD of 0.40 indicates significant deviation from the global distribution.

</details>

<details>
<summary><b>Question 4:</b> Why might t-closeness be too restrictive for practical data publishing? (Click to reveal answer)</summary>

**Answer:** 
**t-Closeness Limitations:**

1. **Over-Suppression:** Requires distributions to closely match global patterns, often leading to excessive generalization or record suppression

2. **Utility Loss:** Many real-world analysis tasks require preserving local patterns that t-closeness deliberately obscures

3. **Implementation Complexity:** Calculating Earth Mover's Distance and choosing appropriate t values is computationally expensive and requires domain expertise

4. **Semantic Issues:** May not make sense for all attribute types (e.g., forcing uniform distribution of rare diseases)

5. **Group Size Requirements:** Often requires very large equivalence classes to achieve acceptable t values, reducing dataset granularity

6. **Analysis Limitations:** Prevents legitimate research that depends on understanding sub-population differences

**Example:** Medical research studying disease prevalence in specific demographics becomes impossible if all groups must mirror global disease distributions.

</details>

<details>
<summary><b>Question 5:</b> Design a scenario where k-anonymity provides sufficient privacy protection and another where l-diversity is necessary. (Click to reveal answer)</summary>

**Answer:** 
**Scenario 1 - k-Anonymity Sufficient:**
**Context:** Survey data about shopping preferences

**Data:** Age, Gender, ZIP → Favorite Store (Amazon, Walmart, Target, Best Buy, etc.)

**Why k-anonymity works:**
- Many diverse store preferences exist
- No sensitive health/financial information
- Natural diversity in consumer choices
- Low stakes if preference is inferred

**k=5 provides adequate protection** for this commercial use case.

**Scenario 2 - l-Diversity Necessary:**
**Context:** Medical insurance claims data

**Data:** Age, Gender, ZIP → Disease (HIV, Cancer, Depression, Common Cold, etc.)

**Why l-diversity needed:**
- High sensitivity of medical information
- Risk of discrimination/stigma
- Potential for homogeneous groups (e.g., all cancer patients in oncology clinic area)
- Background knowledge attacks possible (family/friends know general health status)

**l=3 diversity ensures** each group has at least 3 different conditions, preventing certain medical inference even with background knowledge.

</details>

---

## 📚 Additional Resources

### Core Papers
- Machanavajjhala, A. et al. (2007). "l-diversity: Privacy beyond k-anonymity"
- Li, N. et al. (2007). "t-closeness: Privacy beyond k-anonymity and l-diversity"

### Implementation Guides
- Fung, B. C. M. et al. (2010). "Privacy-preserving data publishing: A survey"
- Gkoulalas-Divanis, A. & Loukides, G. (2012). "Utility-aware anonymization of sets of transactions"

### Tools and Software
- **ARX:** Comprehensive anonymization tool with l-diversity and t-closeness support
- **UTD Anonymization Toolbox:** Research-oriented implementation
- **sdcMicro (R package):** Statistical disclosure control toolkit
---

<div align="center">
  <a href="week2.md">⬅️ <strong>Week 2</strong></a> |
  <a href="README.md"><strong>Main</strong></a> |
  <a href="week4.md"><strong>Week 4</strong> ➡️</a>
</div>
