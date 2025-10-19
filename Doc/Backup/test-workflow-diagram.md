# MP3 to Subtitle Test Workflow Diagram

## Test Workflow Overview

```mermaid
flowchart TD
    A[Start Test] --> B[Check Environment]
    B --> C[Activate srt_env]
    C --> D[Initialize GPU Manager]
    D --> E[Get Audio Files List]
    E --> F{More Files?}
    
    F -->|Yes| G[Load Whisper Large Model]
    G --> H[Transcribe with Whisper]
    H --> I[Generate SRT with Running Number]
    I --> J[Assess Quality Metrics]
    J --> K[Load Typhoon Model]
    K --> L[Transcribe with Typhoon]
    L --> M[Generate SRT with Running Number]
    M --> N[Assess Quality Metrics]
    N --> O[Store Results]
    O --> P[Next Audio File]
    P --> F
    
    F -->|No| Q[Generate Test Report]
    Q --> R[Save test_report.json]
    R --> S[Display Summary]
    S --> T[End Test]
    
    style A fill:#9f9,stroke:#333,stroke-width:2px
    style T fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:1px
    style K fill:#bbf,stroke:#333,stroke-width:1px
    style I fill:#bfb,stroke:#333,stroke-width:1px
    style M fill:#bfb,stroke:#333,stroke-width:1px
    style Q fill:#fbb,stroke:#333,stroke-width:1px
```

## Quality Assessment Flow

```mermaid
flowchart LR
    A[Subtitle Segments] --> B[Check Timestamps]
    B --> C{Valid Format?}
    C -->|No| D[Penalty: -10]
    C -->|Yes| E[Calculate Durations]
    E --> F{Avg Duration 2-7s?}
    F -->|No| G[Penalty: -15]
    F -->|Yes| H[Check Character Count]
    H --> I{Max Chars < 84?}
    I -->|No| J[Penalty: -10]
    I -->|Yes| K[Calculate Quality Score]
    D --> L[Final Score]
    G --> L
    J --> L
    K --> L
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#9f9,stroke:#333,stroke-width:2px
    style D fill:#f99,stroke:#333,stroke-width:1px
    style G fill:#f99,stroke:#333,stroke-width:1px
    style J fill:#f99,stroke:#333,stroke-width:1px
```

## Model Comparison Process

```mermaid
flowchart TB
    subgraph "Audio File Processing"
        A[Input Audio] --> B[Whisper Large]
        A --> C[Typhoon ASR]
    end
    
    subgraph "Output Generation"
        B --> D[whisper.large.001.srt]
        C --> E[typhoon.001.srt]
    end
    
    subgraph "Quality Metrics"
        D --> F[Timing Accuracy]
        D --> G[Segment Duration]
        D --> H[Text Segmentation]
        E --> I[Timing Accuracy]
        E --> J[Segment Duration]
        E --> K[Text Segmentation]
    end
    
    subgraph "Comparison"
        F --> L[Speed Comparison]
        G --> L
        I --> L
        J --> L
        H --> M[Quality Score]
        K --> M
        L --> N[Final Report]
        M --> N
    end
    
    style A fill:#ff9,stroke:#333,stroke-width:2px
    style N fill:#9f9,stroke:#333,stroke-width:2px
```

## File Naming Convention

```mermaid
flowchart LR
    A[Audio File] --> B[Extract Basename]
    B --> C[Add Model Name]
    C --> D[Add Running Number]
    D --> E[Add .srt Extension]
    
    subgraph "Examples"
        F[Jasmali.MP3] --> G[Jasmali.whisper.large.001.srt]
        F --> H[Jasmali.typhoon.001.srt]
        I[ขนมครก.MP3] --> J[ขนมครก.whisper.large.001.srt]
        I --> K[ขนมครก.typhoon.001.srt]
    end
    
    style A fill:#ff9,stroke:#333,stroke-width:2px
    style E fill:#9f9,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:1px
    style H fill:#bbf,stroke:#333,stroke-width:1px
    style J fill:#bbf,stroke:#333,stroke-width:1px
    style K fill:#bbf,stroke:#333,stroke-width:1px