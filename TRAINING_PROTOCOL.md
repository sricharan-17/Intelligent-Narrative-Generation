# Intelligent Narrative Generation
## 3-Model Fine-Tuning Experimental Protocol

**Project:** Intelligent Narrative Generation for Interactive Gaming Using Fine-Tuned Large Language Models

**Experiment Type:** Comparative Fine-Tuning Study

**Fine-Tuning Method:** LoRA / PEFT

**Dataset:** LIGHT

---

# 1. PURPOSE OF THE EXPERIMENT

The purpose of this experiment is to fine-tune three different
base Large Language Models on the same LIGHT dataset and compare
their performance for interactive narrative generation.

The three models will be independently fine-tuned by the three
team members.

The final comparison will evaluate:

1. Action Interpretation
2. Context Relevance
3. Plausibility
4. Narrative Quality
5. State Consistency
6. Response Latency
7. Model Size and Resource Requirements

The comparison must be reproducible and fair.

---

# 2. TEAM MODEL ASSIGNMENT

| Team Member | Base Model | Parameters |
|-------------|------------|------------|
| Member 1 | SmolLM2-1.7B-Instruct | 1.7B |
| Member 2 | Qwen2.5-3B-Instruct | 3B |
| Member 3 | Qwen2.5-7B-Instruct | 7B |

Each member must train only their assigned model.

The base model is the primary experimental variable.

---

# 3. EXPERIMENTAL CONTROL

The following must remain identical across all three models:

- Dataset
- Dataset preprocessing
- Train/validation/test split
- Random seed
- Prompt semantics
- Fine-tuning methodology
- LoRA methodology
- Training objective
- Evaluation test set
- Evaluation methodology
- Evaluation criteria

Only model-specific requirements such as tokenizer/chat-template
handling or unavoidable hardware-specific settings may differ.

Any such difference must be documented.

---

# 4. DATASET

## Source

LIGHT (Learning in a Fantasy Text Adventure)

## Current processed dataset

Total processed examples:

120,492

Speech examples:

97,701

Action examples:

22,791

Original records represented:

9,807

---

# 5. DATASET SPLIT

The dataset is split at the original-record/episode level.

Random seed:

42

## Training

Records:

7,845

Examples:

96,282

## Validation

Records:

980

Examples:

12,023

## Test

Records:

982

Examples:

12,187

## Leakage

Record overlap:

0

No original record is shared between training,
validation, and test sets.

---

# 6. INPUT DISTRIBUTION

## Training

Speech:

77,909 (80.92%)

Action:

18,373 (19.08%)

## Validation

Speech:

9,861 (82.02%)

Action:

2,162 (17.98%)

## Test

Speech:

9,931 (81.49%)

Action:

2,256 (18.51%)

The distributions are sufficiently similar across the three
splits for the planned experiment.

---

# 7. PROCESSED DATA FORMAT

Each training example contains:

- record_id
- turn_id
- input_character
- responder_character
- setting
- world_state
- history
- input
- input_type
- target

The distinction between input character and responder character
must be preserved.

---

# 8. TASK DEFINITION

The model receives:

- Current setting
- Input character
- Input character persona
- Responder character
- Responder character persona
- Current world state
- Available actions
- Conversation history
- Current player/input action
- Input type

The model generates:

- The responder's next response

Conceptually:

GAME STATE
+
CHARACTER INFORMATION
+
CONVERSATION HISTORY
+
PLAYER INPUT
        |
        v
      LLM
        |
        v
RESPONDER RESPONSE

---

# 9. INTERACTION TYPES

The dataset contains two primary input types.

## Speech

Example:

Input:

"A quiet night this evening..."

Target:

"yes it is"

## Action

Example:

Input:

"hug court wizard"

Target:

"Get off of me, you fool! Who gave you permission to touch me!"

Both interaction types must be retained.

---

# 10. PROMPT STRUCTURE

All models must receive the same semantic information.

The common prompt structure is:

### SETTING

Setting name

Setting category

Setting description

Setting background

### INPUT CHARACTER

Character name

Character persona

### RESPONDER CHARACTER

Character name

Character persona

### WORLD STATE

Context

Room objects

Room agents

Object descriptions

Carrying

Wearing

Wielding

Available actions

### HISTORY

Previous interaction turns

### PLAYER INPUT

Current input

### INPUT TYPE

Speech or action

### RESPONSE

Model-generated response

The exact tokenizer/chat-template formatting may differ between
models, but the semantic information must remain the same.

---

# 11. FINE-TUNING METHOD

Fine-tuning method:

LoRA

Library:

PEFT

Training objective:

Causal Language Modeling / Supervised Fine-Tuning

All three models must use the same LoRA methodology.

The following parameters must be agreed upon by the team before
full training:

- LoRA rank (r)
- LoRA alpha
- LoRA dropout
- Target modules
- Learning rate
- Number of epochs
- Maximum sequence length
- Batch size
- Gradient accumulation
- Optimizer
- Learning-rate scheduler
- Warmup
- Precision
- Gradient checkpointing

These values must be recorded before full training.

---

# 12. RANDOMNESS AND REPRODUCIBILITY

Default experiment seed:

42

The seed should be fixed wherever practical.

The following must be recorded:

- Python version
- PyTorch version
- Transformers version
- PEFT version
- Datasets version
- Accelerate version
- Base model revision/version
- Hardware
- Operating system
- Training configuration

---

# 13. HARDWARE

Each member must record their hardware.

Example:

CPU:
[CPU]

GPU:
[GPU or None]

GPU VRAM:
[GB or N/A]

System RAM:
[GB]

Operating System:
[OS]

If a hardware limitation requires a change to batch size,
gradient accumulation, precision, or another training parameter,
the change must be documented.

---

# 14. TRAINING RECORD

## Member

[NAME]

## Model

[MODEL NAME]

## Model Revision

[VERSION / COMMIT / REVISION]

## Hardware

[HARDWARE]

## Software

Python:
[VERSION]

PyTorch:
[VERSION]

Transformers:
[VERSION]

PEFT:
[VERSION]

Datasets:
[VERSION]

Accelerate:
[VERSION]

## Configuration

Epochs:
[VALUE]

Learning Rate:
[VALUE]

Batch Size:
[VALUE]

Gradient Accumulation:
[VALUE]

Maximum Sequence Length:
[VALUE]

LoRA Rank:
[VALUE]

LoRA Alpha:
[VALUE]

LoRA Dropout:
[VALUE]

Target Modules:
[VALUE]

Precision:
[VALUE]

Optimizer:
[VALUE]

Scheduler:
[VALUE]

Warmup:
[VALUE]

Gradient Checkpointing:
[VALUE]

Random Seed:
42

---

# 15. TRAINING RESULTS

Training Start:

[DATE/TIME]

Training End:

[DATE/TIME]

Total Training Time:

[VALUE]

Trainable Parameters:

[VALUE]

Total Parameters:

[VALUE]

Final Training Loss:

[VALUE]

Best Validation Loss:

[VALUE]

Best Checkpoint:

[PATH]

Peak Memory Usage:

[VALUE]

---

# 16. MODEL EVALUATION

Every model must be evaluated using the SAME test set.

The test set must never be modified after the experiment begins.

Primary evaluation categories:

## 16.1 Action Interpretation

Question:

Did the model understand what the player intended?

Measure whether the generated response appropriately reflects
the supplied speech/action input.

---

## 16.2 Context Relevance

Question:

Did the model use the supplied game context?

Check whether the generated response is relevant to the current
location, characters, objects, and situation.

---

## 16.3 Plausibility

Question:

Did the model produce a sensible response given the available
actions and current situation?

---

## 16.4 Narrative Quality

Question:

Is the generated continuation coherent and appropriate as an
interactive narrative?

---

## 16.5 State Consistency

Question:

Does the response respect the current game state?

Check:

- Characters
- Objects
- Inventory
- Location
- Available actions
- Previous events

---

## 16.6 Response Latency

Measure:

- Generation time
- Average response time
- Relevant hardware conditions

The same inference conditions should be used where practical.

---

## 16.7 Model Size and Resources

Record:

- Parameter count
- Model storage size
- Trainable parameter count
- Peak RAM/VRAM
- Inference resource usage
- Training time

---

# 17. QUALITATIVE EVALUATION

The same scenarios must be evaluated across all three models.

For each selected scenario record:

Scenario ID:

[ID]

Game State:

[STATE]

Input Character:

[CHARACTER]

Responder Character:

[CHARACTER]

Player Input:

[INPUT]

Input Type:

[SPEECH / ACTION]

Reference Target:

[TARGET]

Model Output:

[OUTPUT]

Observations:

[OBSERVATIONS]

The same scenario IDs must be used for every model.

---

# 18. MODEL COMPARISON

The final results table should contain:

| Capability | Model 1 | Model 2 | Model 3 |
|------------|---------|---------|---------|
| Action Interpretation | | | |
| Context Relevance | | | |
| Plausibility | | | |
| Narrative Quality | | | |
| State Consistency | | | |
| Response Latency | | | |
| Model Size | | | |
| Resource Usage | | | |
| Training Time | | | |

No model should be evaluated using a different test set.

---

# 19. EXPERIMENT CHANGE LOG

Every significant change must be recorded.

## Entry

Date:

[DATE]

Member:

[NAME]

Model:

[MODEL]

Change:

[DESCRIPTION]

Reason:

[REASON]

Effect:

[RESULT]

Approved by team:

[YES / NO]

---

# 20. AI ASSISTANT USAGE RULES

All team members may use AI assistants such as ChatGPT or Claude
for implementation assistance.

However:

1. The AI must be given this protocol.
2. The AI must know the assigned model.
3. The AI must not redesign the experiment without team approval.
4. AI-generated code must be tested before being accepted.
5. Terminal output must be checked before declaring a step successful.
6. Training configuration changes must be recorded.
7. AI must not silently modify the dataset or test split.
8. If an AI recommends changing a shared experimental parameter,
   the team must discuss and approve the change.
9. Errors must be documented when they affect the experiment.
10. The final experimental configuration must be reproducible.

---

# 21. AI ASSISTANT MASTER INSTRUCTION

When asking an AI assistant for help, provide:

"You are helping me implement the fine-tuning component of our
Final Year Project.

Project:
Intelligent Narrative Generation for Interactive Gaming Using
Fine-Tuned Large Language Models.

My assigned model:
[MODEL]

Follow the attached/shared TRAINING_PROTOCOL.md exactly.

I am a beginner in LLM fine-tuning.

Do not change the dataset, train/validation/test split,
evaluation methodology, or experimental design without
explicit team approval.

Explain each step before asking me to execute it.

Do not claim that a step succeeded until I provide the actual
terminal output.

If an error occurs, diagnose the actual error first.

Keep all implementation decisions reproducible.

Current project directory:

D:\Final_Year_Project\Intelligent_Narrative_Generation

Current dataset:

data/processed/light_processed.jsonl

Current splits:

data/splits/train.jsonl
data/splits/validation.jsonl
data/splits/test.jsonl
"

---

# 22. EXPERIMENTAL PRINCIPLE

The goal is not simply to determine which model produces the
most fluent text.

The experiment studies the relationship between model capacity,
resource requirements, and interactive narrative capabilities.

The final analysis must consider the complete set of measured
results:

Action Interpretation
Context Relevance
Plausibility
Narrative Quality
State Consistency
Response Latency
Model Size
Resource Requirements

All conclusions must be supported by the recorded experimental
results.

---

# 23. CURRENT PROJECT STATUS

Completed:

[X] LIGHT dataset acquisition

[X] Structured LIGHT inspection

[X] Dataset statistics

[X] Preprocessing

[X] Processed dataset validation

[X] Record-level train/validation/test split

[X] Split leakage check

[X] Split distribution check

[X] Input/responder character distinction

[X] Action example validation

Current stage:

[ ] Training environment setup

[ ] Model loading test

[ ] Tokenizer/chat-template test

[ ] Training smoke test

[ ] Full LoRA training

[ ] Model evaluation

[ ] Three-model comparison

---

# 24. DATASET FREEZE

The following files constitute the current dataset version:

data/processed/light_processed.jsonl

data/splits/train.jsonl

data/splits/validation.jsonl

data/splits/test.jsonl

Dataset split seed:

42

These files must not be modified during the three-model
comparison experiment.

If a dataset modification becomes necessary, the team must:

1. Document the reason.
2. Create a new dataset version.
3. Regenerate all splits.
4. Re-run validation.
5. Restart the controlled experiment.

---

# END OF PROTOCOL