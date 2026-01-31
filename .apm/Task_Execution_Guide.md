# APM Task Execution Guide

This guide outlines the manual task execution process for the Price Movement Probability Discovery System project.

## Task Execution Process

The APM system uses a manual task execution workflow:

1. **Assign Task** - Send task assignment prompt to the appropriate agent
2. **Wait for Completion** - Monitor memory logs for task completion
3. **Review Results** - Examine the completed memory log and generated artifacts
4. **Assign Next Task** - Based on dependencies, assign the next appropriate task

## Phase 2 Task Sequence (Completed)

### Task 2.1 - Data Acquisition Pipeline Enhancement
- **Agent**: Agent_DataEngineering
- **Dependencies**: None
- **Assignment File**: .apm/Task_Assignment_Prompts/Task_2_1_Data_Acquisition_Pipeline_Enhancement.md
- **Memory Log**: .apm/Memory/Phase_02_Data_Pipeline_Enhancement/Task_2_1_Data_Acquisition_Pipeline_Enhancement.md

### Tasks 2.2 & 2.3 (Executed in parallel after Task 2.1 completion)
- **Task 2.2**: Feature Engineering Optimization
  - **Agent**: Agent_DataEngineering
  - **Dependencies**: Task 2.1 Output
  - **Assignment File**: .apm/Task_Assignment_Prompts/Task_2_2_Feature_Engineering_Optimization.md
  - **Memory Log**: .apm/Memory/Phase_02_Data_Pipeline_Enhancement/Task_2_2_Feature_Engineering_Optimization.md

- **Task 2.3**: Technical Indicator Library Development
  - **Agent**: Agent_DataEngineering
  - **Dependencies**: Task 2.1 Output
  - **Assignment File**: .apm/Task_Assignment_Prompts/Task_2_3_Technical_Indicator_Library_Development.md
  - **Memory Log**: .apm/Memory/Phase_02_Data_Pipeline_Enhancement/Task_2_3_Technical_Indicator_Library_Development.md

## Phase 3 Task Sequence (Completed)

### Tasks 3.1 & 3.2 (Executed in parallel)
- **Task 3.1**: Machine Learning Pattern Discovery Implementation
  - **Agent**: Agent_PatternDiscovery
  - **Dependencies**: Task 2.2 Output by Agent_DataEngineering
  - **Assignment File**: .apm/Task_Assignment_Prompts/Task_3_1_Machine_Learning_Pattern_Discovery_Implementation.md
  - **Memory Log**: .apm/Memory/Phase_03_Pattern_Discovery_Implementation/Task_3_1_Machine_Learning_Pattern_Discovery_Implementation.md

- **Task 3.2**: Rule-Based Pattern Enhancement
  - **Agent**: Agent_PatternDiscovery
  - **Dependencies**: Task 2.2 Output by Agent_DataEngineering
  - **Assignment File**: .apm/Task_Assignment_Prompts/Task_3_2_Rule_Based_Pattern_Enhancement.md
  - **Memory Log**: .apm/Memory/Phase_03_Pattern_Discovery_Implementation/Task_3_2_Rule_Based_Pattern_Enhancement.md

### Task 3.3 (Depends on completion of both Tasks 3.1 and 3.2)
- **Task 3.3**: Pattern Validation Framework Development
  - **Agent**: Agent_Validation
  - **Dependencies**: Task 3.1 Output by Agent_PatternDiscovery and Task 3.2 Output by Agent_PatternDiscovery
  - **Assignment File**: .apm/Task_Assignment_Prompts/Task_3_3_Pattern_Validation_Framework_Development.md
  - **Memory Log**: .apm/Memory/Phase_03_Pattern_Discovery_Implementation/Task_3_3_Pattern_Validation_Framework_Development.md

## Phase 4 Task Sequence (Current Phase)

### Task 4.1 - Pattern Performance Testing
- **Agent**: Agent_Validation
- **Dependencies**: Task 3.3 Output
- **Assignment File**: .apm/Task_Assignment_Prompts/Task_4_1_Pattern_Performance_Testing.md
- **Memory Log**: .apm/Memory/Phase_04_Validation_Testing/Task_4_1_Pattern_Performance_Testing.md

### Task 4.2 - False Positive Reduction Validation
- **Agent**: Agent_Validation
- **Dependencies**: Task 3.2 Output by Agent_PatternDiscovery
- **Assignment File**: .apm/Task_Assignment_Prompts/Task_4_2_False_Positive_Reduction_Validation.md
- **Memory Log**: .apm/Memory/Phase_04_Validation_Testing/Task_4_2_False_Positive_Reduction_Validation.md

### Task 4.3 - Cross-Validation Implementation
- **Agent**: Agent_Validation
- **Dependencies**: Task 4.1 Output
- **Assignment File**: .apm/Task_Assignment_Prompts/Task_4_3_Cross_Validation_Implementation.md
- **Memory Log**: .apm/Memory/Phase_04_Validation_Testing/Task_4_3_Cross_Validation_Implementation.md

## Verification Process

After each task completion:
1. Check the memory log for completion status
2. Verify that expected artifacts were created
3. Review any important findings or compatibility issues
4. Proceed to next task based on dependencies

## Next Phase Preparation

After all Phase 4 tasks are complete:
1. Create Phase 5 directory: .apm/Memory/Phase_05_Visualization_Monitoring/
2. Create empty memory log files for Phase 5 tasks
3. Prepare Task Assignment Prompts for Phase 5 tasks