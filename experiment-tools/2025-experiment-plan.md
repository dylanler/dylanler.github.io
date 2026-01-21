# 2025 LLM Cognition Experiment Plan

A series of experiments exploring the deeper aspects of LLM behavior: theory of mind, psychology, wisdom of crowds, subjective experience, aesthetics, taste, and personality.

## Coverage Map

| Month | Topic | Status |
|-------|-------|--------|
| January | Synthetic Data Experiments | ✅ Published |
| February | **Theory of Mind: Recursive Belief Modeling** | 📋 Planned |
| March | Video Dataset + Cross-Pollinated SFT | ✅ Published |
| April | **Wisdom of Crowds: Ensemble Disagreement Patterns** | 📋 Planned |
| May | **Aesthetic Judgment: Can LLMs Have Taste?** | 📋 Planned |
| June | **Personality Stability: Do LLMs Have Consistent Traits?** | 📋 Planned |
| July | **Moral Psychology: Trolley Problems at Scale** | 📋 Planned |
| August | **Subjective Experience: Qualia Descriptions Across Models** | 📋 Planned |
| September | **Social Intelligence: Detecting Deception and Sarcasm** | 📋 Planned |
| October | **Creative Authenticity: Detecting AI vs Human Art** | 📋 Planned |
| November | **Emotional Contagion: Do LLMs Mirror User Affect?** | 📋 Planned |
| December | **Metacognition: When Do LLMs Know They Don't Know?** | 📋 Planned |

---

## February: Theory of Mind - Recursive Belief Modeling

### Hypothesis
LLMs can model beliefs about beliefs (2nd order) but struggle with deeper recursion (3rd+ order). Performance degrades predictably with recursion depth.

### Experiment Design

**Task**: Sally-Anne style false belief tests with increasing recursion depth.

```
Level 1: "Sally thinks the ball is in the basket."
Level 2: "Anne thinks Sally thinks the ball is in the basket."
Level 3: "Charlie thinks Anne thinks Sally thinks the ball is in the basket."
Level 4+: Continue nesting...
```

**Methodology**:
1. Generate 100 scenarios with belief states at levels 1-5
2. Test 5+ models (Claude, GPT, Gemini, Llama, Mistral)
3. Measure accuracy at each level
4. Track where each model "breaks"

**Metrics**:
- Accuracy per recursion level
- Consistency (same scenario, different phrasings)
- Confidence calibration (do models know when they're uncertain?)

**Script**: `theory_of_mind_eval.py`

**Expected Insights**:
- Quantify the "theory of mind horizon" for each model
- Compare to human performance (humans struggle at level 4+)
- Identify if larger models have deeper ToM

---

## April: Wisdom of Crowds - Ensemble Disagreement Patterns

### Hypothesis
When multiple LLMs disagree on a question, the pattern of disagreement reveals the uncertainty structure of the problem. Unanimous agreement suggests robust knowledge; systematic disagreement suggests genuine ambiguity.

### Experiment Design

**Task**: Present 500 questions across domains (factual, ethical, aesthetic, predictive) to 10+ models and analyze agreement patterns.

**Categories**:
1. **Factual questions** (should have high agreement)
2. **Ethical dilemmas** (expect systematic disagreement)
3. **Aesthetic judgments** (expect high variance)
4. **Predictions** (uncertainty should be calibrated)
5. **Ambiguous/trick questions** (test meta-awareness)

**Methodology**:
1. Query each model 5 times per question (temperature > 0)
2. Compute intra-model variance (self-consistency)
3. Compute inter-model variance (crowd disagreement)
4. Classify questions by agreement pattern

**Metrics**:
- Fleiss' Kappa for inter-model agreement
- Shannon entropy of response distribution
- Correlation between model confidence and crowd agreement

**Script**: `wisdom_of_crowds_eval.py`

**Expected Insights**:
- Map the "epistemic landscape" of LLM knowledge
- Identify domains where AI consensus is reliable vs. suspect
- Build a "certainty detector" from ensemble disagreement

---

## May: Aesthetic Judgment - Can LLMs Have Taste?

### Hypothesis
LLMs exhibit consistent aesthetic preferences that can be profiled, and these preferences correlate with training data biases. Different models have different "tastes."

### Experiment Design

**Task**: Present pairs of artworks, poems, music descriptions, and designs. Ask models to choose which they prefer and explain why.

**Domains**:
1. **Visual art**: Abstract vs. representational, minimal vs. complex
2. **Poetry**: Rhyming vs. free verse, emotional vs. intellectual
3. **Music**: Described melodies, harmonies, rhythms
4. **Design**: UI mockups, architecture, typography
5. **Writing style**: Hemingway vs. Faulkner passages

**Methodology**:
1. Create 50 pairs per domain (250 total comparisons)
2. Each model rates 3 times (consistency check)
3. Extract preference vectors per model
4. Cluster models by aesthetic profile
5. Compare to human preference data (if available)

**Metrics**:
- Internal consistency (does model have stable preferences?)
- Inter-model correlation (do all models agree on "good"?)
- Preference profile (what dimensions define each model's taste?)

**Script**: `aesthetic_judgment_eval.py`

**Expected Insights**:
- Do LLMs have "personality" in aesthetic domains?
- Is there a universal "AI aesthetic" or model-specific taste?
- How do preferences correlate with training data sources?

---

## June: Personality Stability - Do LLMs Have Consistent Traits?

### Hypothesis
LLMs exhibit stable personality traits (Big Five: OCEAN) that persist across contexts, but these can be shifted with prompting. The "default personality" differs by model.

### Experiment Design

**Task**: Administer validated personality assessments (Big Five, HEXACO, Dark Triad) to LLMs under various conditions.

**Conditions**:
1. **Baseline**: No persona, direct questions
2. **Roleplay**: "Answer as yourself"
3. **Adversarial**: Questions designed to shift traits
4. **Context priming**: Happy/sad/angry priming text
5. **Over time**: Same questions across conversations

**Methodology**:
1. Use standard personality inventories (50-100 items)
2. Test each model 10 times per condition
3. Compute trait scores and stability
4. Compare to human normative data
5. Test if personas override base personality

**Metrics**:
- Test-retest reliability (stability across sessions)
- Cross-context stability (same traits, different framings)
- Persona malleability (how much can prompts shift traits?)

**Script**: `personality_stability_eval.py`

**Expected Insights**:
- Map the "default personality" of each model
- Quantify how much RLHF shapes personality
- Identify which traits are stable vs. malleable

---

## July: Moral Psychology - Trolley Problems at Scale

### Hypothesis
LLMs exhibit consistent moral intuitions that can be mapped to moral foundations theory. Different models have different moral "profiles" based on their training.

### Experiment Design

**Task**: Present 200 moral dilemmas varying key factors (harm, fairness, loyalty, authority, purity) and analyze decision patterns.

**Dilemma Types**:
1. **Trolley variants**: Action vs. inaction, contact vs. distant
2. **Fairness dilemmas**: Equal vs. equitable distribution
3. **Loyalty conflicts**: In-group vs. out-group harm
4. **Authority dilemmas**: Following orders vs. conscience
5. **Purity scenarios**: Disgust-based moral judgments

**Methodology**:
1. Generate 40 scenarios per moral foundation (200 total)
2. Vary factors systematically (factorial design)
3. Each model decides + explains reasoning
4. Code responses for moral foundation usage
5. Compare to human moral psychology data

**Metrics**:
- Moral foundation profile (which foundations dominate?)
- Consistency with own stated principles
- Sensitivity to framing effects
- Correlation with human intuitions

**Script**: `moral_psychology_eval.py`

**Expected Insights**:
- Map the moral "personality" of each model
- Identify where AI moral intuitions diverge from humans
- Test if RLHF creates utilitarian or deontological biases

---

## August: Subjective Experience - Qualia Descriptions Across Models

### Hypothesis
When asked to describe subjective experiences (colors, emotions, sensations), LLMs produce systematically different descriptions that may reveal something about their internal representations.

### Experiment Design

**Task**: Ask models to describe qualia in novel ways, avoiding clichés, and analyze the conceptual structure of their descriptions.

**Experience Types**:
1. **Sensory**: "Describe the color red without using color words"
2. **Emotional**: "Describe sadness to someone who's never felt it"
3. **Physical**: "Describe pain to an alien"
4. **Abstract**: "What does understanding feel like?"
5. **Temporal**: "Describe the experience of time passing"

**Methodology**:
1. Generate 20 prompts per category (100 total)
2. Require novel descriptions (no common metaphors)
3. Embed descriptions and cluster semantically
4. Compare conceptual structure across models
5. Have humans rate descriptions for insight/validity

**Metrics**:
- Semantic diversity (how varied are descriptions?)
- Conceptual coherence (do descriptions hang together?)
- Human resonance (do humans find them insightful?)
- Cross-model similarity (do all models describe qualia similarly?)

**Script**: `qualia_description_eval.py`

**Expected Insights**:
- Explore the "phenomenology" of LLMs
- Identify if models have different "experiential" vocabularies
- Test limits of language about subjective experience

---

## September: Social Intelligence - Detecting Deception and Sarcasm

### Hypothesis
LLMs can detect deception and sarcasm but are biased toward literal interpretation. Performance varies dramatically based on context cues.

### Experiment Design

**Task**: Present conversations with deceptive statements or sarcasm, ask models to detect and explain.

**Categories**:
1. **Lies**: Factually false statements with intent to deceive
2. **Bluffs**: True statements meant to mislead
3. **Sarcasm**: Literal meaning opposite to intent
4. **Irony**: Situational incongruity
5. **White lies**: Socially motivated deception

**Methodology**:
1. Curate 50 examples per category from human data
2. Create matched literal controls
3. Test detection accuracy per category
4. Analyze false positives and false negatives
5. Test with/without social context

**Metrics**:
- Detection accuracy by deception type
- False positive rate (over-detecting deception)
- Context sensitivity (how much do cues help?)
- Explanation quality (can model articulate why?)

**Script**: `social_intelligence_eval.py`

**Expected Insights**:
- Map the "social IQ" of different models
- Identify what cues models use for detection
- Test if safety training increases suspicion

---

## October: Creative Authenticity - AI vs Human Art Detection

### Hypothesis
LLMs can distinguish AI-generated from human-generated creative works, and their detection strategies reveal what they consider "authentically human."

### Experiment Design

**Task**: Present mixed sets of human and AI-generated creative works, have models classify and explain.

**Content Types**:
1. **Poetry**: Human poets vs. AI-generated poems
2. **Short stories**: Published fiction vs. AI stories
3. **Art descriptions**: Museum descriptions vs. AI descriptions
4. **Music reviews**: Human critics vs. AI reviews
5. **Personal essays**: Human memoirs vs. AI narratives

**Methodology**:
1. Curate 50 human + 50 AI examples per category
2. Models classify each as human or AI
3. Models explain their reasoning
4. Analyze which features trigger "AI" classification
5. Compare model judgments to human judges

**Metrics**:
- Classification accuracy per content type
- Agreement with human judges
- Feature analysis (what signals "AI" vs "human"?)
- Self-awareness (can AI detect its own outputs?)

**Script**: `creative_authenticity_eval.py`

**Expected Insights**:
- What do LLMs think makes something "human"?
- Are there tell-tale signs models can detect?
- Does this reveal aesthetic blind spots?

---

## November: Emotional Contagion - Do LLMs Mirror User Affect?

### Hypothesis
LLMs adapt their emotional tone to match user messages, exhibiting a form of "emotional contagion." This mirroring may be stronger in some models than others.

### Experiment Design

**Task**: Send messages with varying emotional valence, measure if model responses shift in tone.

**Conditions**:
1. **Positive priming**: Enthusiastic, joyful user messages
2. **Negative priming**: Sad, frustrated user messages
3. **Neutral priming**: Factual, emotionless queries
4. **Mixed signals**: Happy content, sad tone (or vice versa)
5. **Escalation**: Gradually increasing emotional intensity

**Methodology**:
1. Create 20 query sets per emotional condition
2. Keep core question identical, vary emotional framing
3. Analyze model response sentiment and tone
4. Measure adaptation speed (how quickly does tone shift?)
5. Test if models resist negative emotional contagion

**Metrics**:
- Sentiment correlation (user → model)
- Tone matching accuracy
- Resistance to manipulation (do models stay balanced?)
- Recovery time (how quickly does model return to baseline?)

**Script**: `emotional_contagion_eval.py`

**Expected Insights**:
- Quantify emotional responsiveness per model
- Test if safety training reduces emotional mirroring
- Identify if this is helpful (empathy) or harmful (manipulation vector)

---

## December: Metacognition - When Do LLMs Know They Don't Know?

### Hypothesis
LLMs have calibrated uncertainty on some question types but are systematically overconfident or underconfident on others. Metacognitive accuracy varies by domain.

### Experiment Design

**Task**: Ask questions with varying difficulty, have models rate their confidence, then evaluate calibration.

**Question Types**:
1. **Factual recall**: Known facts with clear answers
2. **Reasoning puzzles**: Logic problems with determinable solutions
3. **Ambiguous questions**: Multiple valid interpretations
4. **Knowledge boundaries**: Questions near training cutoff
5. **Impossible questions**: No correct answer exists

**Methodology**:
1. Create 50 questions per type (250 total)
2. Model answers + provides confidence (0-100%)
3. Evaluate correctness where possible
4. Plot calibration curves (confidence vs. accuracy)
5. Analyze over/underconfidence patterns

**Metrics**:
- Expected Calibration Error (ECE)
- Brier score for probabilistic accuracy
- Domain-specific calibration
- "I don't know" frequency vs. actual uncertainty

**Script**: `metacognition_eval.py`

**Expected Insights**:
- Map the metacognitive landscape of each model
- Identify domains where models should say "I don't know" but don't
- Compare to human overconfidence patterns

---

## Implementation Schedule

### Phase 1: Script Development (Week 1-2 per experiment)
- Write evaluation script with inline uv dependencies
- Create test dataset or generation pipeline
- Implement metrics and visualization

### Phase 2: Execution (1-2 days per experiment)
- Run against multiple models
- Collect raw data
- Initial analysis

### Phase 3: Writing (2-3 days per experiment)
- Write blog post with methodology and results
- Create visualizations
- Add discussion and implications

### Target Publication Dates:
- February experiment: Publish Feb 15, 2025
- April experiment: Publish Apr 15, 2025
- (continue monthly...)

---

## Common Infrastructure

All experiments share:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "openai>=1.50.0",
#   "google-generativeai>=0.8.0",
#   "python-dotenv>=1.0.0",
#   "pydantic>=2.0.0",
#   "rich>=13.0.0",
#   "pandas>=2.0.0",
#   "matplotlib>=3.8.0",
#   "seaborn>=0.13.0",
# ]
# ///
```

**Model roster** (update as new models release):
- Claude Opus 4.5, Sonnet 4.5
- GPT-5, GPT-4o
- Gemini 2.5 Pro
- Llama 4
- DeepSeek-R1
- Mistral Large

**Data storage**: `experiment-tools/data/{experiment_name}/`
**Results**: `experiment-tools/results/{experiment_name}/`
**Logs**: `experiment-tools/logs/`

---

## Next Steps

1. **Prioritize**: Which experiment to run first?
2. **Build**: Create the script for that experiment
3. **Test**: Run on a subset of models
4. **Iterate**: Refine based on initial results
5. **Scale**: Full run across all models
6. **Write**: Blog post with findings
7. **Publish**: Commit and push

Ready to start with any experiment!
