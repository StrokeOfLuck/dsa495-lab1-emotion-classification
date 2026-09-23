# Lab 1: Emotion Classification and Error Analysis

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StrokeOfLuck/dsa495-lab1-emotion-classification/blob/main/Lab1.ipynb)  
[View the notebook and its saved outputs on GitHub](https://github.com/StrokeOfLuck/dsa495-lab1-emotion-classification/blob/main/Lab1.ipynb)

The repository is private. If Colab asks, authorize access to your GitHub account; running the notebook also needs `emotion.csv` in your Google Drive.

**Student:** Sean Ryan

## Purpose

This task assigns one of six emotion labels to each message in the supplied dataset. Comparing a constant joy prediction, a DistilBERT checkpoint trained for emotion classification, and a BART natural-language-inference checkpoint used with zero-shot candidate labels shows how task-specific training and label wording affect performance on the same held-out messages.

## Files and rerun instructions

- `Lab1.ipynb`: code, displayed tables and figures for the split, tokenizer inspection, model predictions, evaluation, and error analysis.
- `README.md`: written responses and reproducibility instructions.
- No additional submission files. The course dataset is already provided separately.

To run the analysis:

1. Open the notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Mount Google Drive when prompted and put `emotion.csv` in `MyDrive/Text analysis for Data Science/Lab 1` (or edit `COLAB_LAB_DIR` to its actual location).
3. Run all cells from top to bottom. The first cell installs the pinned `transformers` version; the first model run downloads the two pinned checkpoints. Keep the random seed and split unchanged to reproduce the comparisons. In a local run, place `emotion.csv` beside the notebook instead.

## 1. Data and tokenization

### Q1. Development and evaluation data

**Response:** The seed-495 sample has 30 development messages, five from each emotion, and 1,970 evaluation messages. Surprise is least common in evaluation (61 messages, 3.10%), while joy has 690 (35.03%). Predicting only joy therefore reaches 35.03% accuracy while recognizing none of the other five classes; macro-F1 and per-class recall help expose that failure.

### Q2. What do the tokenizers receive?

**Response:** In I3, DistilBERT lowercases `HAPPY` to `happy`, making its seven content tokens identical to I4; BART preserves the capitalized spelling and splits it into `ĠH`, `APP`, `Y` (nine versus seven content tokens). In my S1 (`I am NOT okay with this result 😟.`), DistilBERT lowercases `NOT` and maps the emoji to `[UNK]`, whereas BART retains `ĠNOT` and represents the emoji with byte-level pieces. My S2 also splits `bittersweet` differently: DistilBERT uses `bitter`, `##sw`, `##eet`; BART uses `Ġbitters`, `weet`. Both add two special tokens to each of these examples.

### Q3. Truncation

**Response:** With the artificial 32-token limit, both tokenizers stop partway through the second repetition of the ordinary train-ride description. The omitted suffix contains the entire decisive contrast, “Despite the ordinary journey, I am terrified about what happens tomorrow.” Removing `terrified` could hide strong evidence for fear, though this demonstration does not establish an actual classification error.

## 2. Specialized encoder classification

### Q4. Baseline and encoder results

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | 0.3503 | 0.0865 | N/A |
| DistilBERT emotion classifier | 0.9244 | 0.8803 | 22.82 |

**Response:** DistilBERT substantially exceeds the constant baseline on both metrics. Its lowest recall is for surprise: 0.7541 over 61 messages (46 correct), so aggregate accuracy would conceal the weaker performance on this rare class.

### Q5. Three encoder errors

| Example ID | Reference label | Prediction | Model score | Brief observation |
|---|---|---|---:|---|
| emotion_test_01314 | surprise | fear | 0.9988 | “feel strange” is underspecified. |
| emotion_test_01377 | love | joy | 0.9986 | “overjoyed” and “beloved friends” suggest two emotions. |
| emotion_test_01270 | joy | sadness | 0.9978 | “very saddened” contradicts the reference label. |

**Response:** These high scores are the model's confidence in its predicted class within its own output, not proof that it is correct or well calibrated. “Feel strange” offers little context for deciding between surprise and fear. “Overjoyed” is explicit joy evidence, while “beloved friends” could support love; a message can express more than one emotion despite the single reference label. The joy reference for “very saddened” appears especially debatable, so treating every disagreement as a clear model failure would overstate the evidence.

## 3. Zero-shot classification

### Q6. Label wording

| Candidate-label formulation | Accuracy | Macro-F1 |
|---|---:|---:|
| A: emotion names | 0.5000 | 0.4644 |
| B: expanded descriptions | 0.5667 | 0.5523 |

**Response:** The prespecified higher-macro-F1 rule selected B, the expanded descriptions (0.5523 versus 0.4644). For `emotion_test_00332`, “feel humiliated” changed from surprise under A to sadness under B, matching the reference. Because the development set contains only five messages per class, a few examples can swing macro-F1 and the chosen wording may not be the best formulation on new data.

### Q7. Final model comparison

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | 0.3503 | 0.0865 | N/A |
| DistilBERT emotion classifier | 0.9244 | 0.8803 | 22.82 |
| BART zero-shot classifier | 0.5365 | 0.4795 | 1085.14 |

**Response:** DistilBERT exceeds BART by 0.3878 in accuracy and 0.4008 in macro-F1 on these messages. BART still exceeds the constant baseline on both metrics. These checkpoints differ in training, task, size, and inference procedure, so this is an operational comparison of the supplied approaches rather than an isolated test of encoder versus encoder-decoder architecture; timings are from this CPU run and exclude model loading.

### Q8. Four model disagreements

| Example ID | Reference | DistilBERT | BART | Who is correct? |
|---|---|---|---|---|
| emotion_test_00002 | sadness | sadness | love | DistilBERT |
| emotion_test_00072 | surprise | fear | surprise | BART |
| emotion_test_00098 | anger | fear | sadness | Neither |
| emotion_test_00004 | sadness | sadness | surprise | DistilBERT |

**Response:** In `emotion_test_00072`, “feels weird” offers BART a plausible surprise cue, while unfamiliar bodily coordination may have led DistilBERT toward fear; the reference favors BART, though the message does not explicitly name an emotion. In `emotion_test_00098`, “heart is tortured by what i have done” supports DistilBERT's fear and BART's sadness or guilt readings, but it does not plainly express the reference anger. The “ashamed” wording in `emotion_test_00002` supports sadness despite its relational context, while the brief “vain” message in `emotion_test_00004` leaves little context. Across all disagreements, DistilBERT alone is correct in 812, BART alone in 48, and neither in 44; these cases warrant reading the text, not merely counting labels.

### Q9. Recommendation and limitations

**Response:** I would use the specialized DistilBERT classifier for this fixed six-emotion task: its 0.9244 accuracy and 0.8803 macro-F1 exceed BART's 0.5365 and 0.4795, and it ran in 22.82 rather than 1085.14 seconds on this CPU. The 812 versus 48 single-model wins reinforce that choice, although BART recognizes the “feels weird” surprise example that DistilBERT misses. The labeled data contain debatable single-emotion references, such as joy for “very saddened”; these metrics cannot establish correctness for nuanced or multiple emotions. The evaluation uses one source test split and one tiny development selection, so it does not establish performance on other domains, new labels, or alternative prompt and checkpoint choices. Model scores were not calibrated here, and the runtime comparison excludes checkpoint loading and depends on hardware.

## AI-use statement

I used ChatGPT (Codex) to draft the two diagnostic examples, complete the notebook's student code blocks, execute the analysis, and draft the interpretations in this README. Its output was checked against the supplied dataset, the notebook's displayed metrics and error rows, and the messages quoted above. The two examples are AI drafted and should be replaced with my own examples before submission if the requirement for student-written examples means independently authored text; I should rerun the notebook and update Q2 after replacing them.
