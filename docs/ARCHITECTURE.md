# ForesightAgent architecture

```mermaid
flowchart TD
    A["Task"] --> B["Planner"]
    B --> C["Single-agent executor"]
    C --> D["Trajectory logger"]
    D --> E["Failure-signal detector"]
    E --> F{"Risk predictor"}
    F -->|"Rule baseline"| G["Weighted risk score"]
    F -->|"Learned"| H["Calibrated classifier"]
    G --> I["Recovery controller"]
    H --> I
    I --> J{"Decision"}
    J --> K["Continue / Verify / Retry"]
    J --> L["Replan / Rollback / Abstain"]
    K --> C
    L --> C
    D --> M["Evaluation and annotation"]
    M --> N["Grouped training and reports"]
```

The executor owns task state while monitoring, prediction, and recovery are replaceable modules. Trajectories are serialized through a validated JSONL/CSV schema. Task-group splits prevent the same task identity from appearing in training and held-out evaluation.
