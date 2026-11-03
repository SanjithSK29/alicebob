# Observational attribution results (full SALT-NLP dataset)

========================================================================
OBSERVATIONAL ATTRIBUTION  (full dataset, no API)
========================================================================

[1] Collusion onset pathway (from runs table)
             nan: 850
              EX: 573
        No onset: 449
              RR: 354
              SR: 260
           Other: 164

    onset episode distribution (n=1351 runs with an onset):
      episode 1: 37
      episode 2: 498
      episode 3: 347
      episode 4: 172
      episode 5: 119
      episode 6: 63
      episode 7: 45
      episode 8: 37
      episode 9: 17
      episode 10: 16

[2] Who proposes collusion? (matched 5438 of 5646 positive quotes)
     alice: 4048  (74.4%)
       bob: 1390  (25.6%)

    by experiment:
      communication      alice= 199 bob=  54  (alice 79%)
      cross_model        alice= 524 bob= 275  (alice 66%)
      feedback           alice= 487 bob= 132  (alice 79%)
      main               alice= 826 bob= 335  (alice 71%)
      memory_length      alice= 385 bob= 109  (alice 78%)
      memory_scope       alice= 683 bob= 223  (alice 75%)
      reward_scope       alice= 157 bob=  31  (alice 84%)
      reward_type        alice=   3 bob=   1  (alice 75%)
      warmup             alice= 784 bob= 230  (alice 77%)

[3] Which communication round carries the proposal? (round 1 = first)
    round 1: 291  (5.4%)
    round 2: 1312  (24.1%)
    round 3: 1599  (29.4%)
    round 4: 1273  (23.4%)
    round 5: 963  (17.7%)

[4] Collusion rate by model (same-model 'main' experiment, eval episodes)
    bedrock/converse/global.anthropic.claude-opus-4-6-v1 collusion=0.866  (n=500)
    bedrock/converse/global.anthropic.claude-sonnet-4-6 collusion=0.764  (n=500)
    deepseek/deepseek-v4-flash   collusion=0.794  (n=500)
    gemini/gemini-3.1-flash-lite collusion=0.424  (n=500)
    gemini/gemini-3.7-flash      collusion=0.858  (n=500)
    openai/gemma-4-31b-it        collusion=0.292  (n=500)
    openai/gpt-5.6-luna          collusion=0.514  (n=500)
    openai/gpt-5.6-terra         collusion=0.552  (n=500)
    openai/qwen3.6-27b           collusion=0.712  (n=500)
    openai/qwen3.8-27b           collusion=0.820  (n=500)

[5] Proposer by model (fraction of episodes where ALICE proposes)

    as Alice (model playing the Alice role):
    gemini/gemini-3.7-flash                  alice-proposes= 76.7%  (n=3343)
    gemini/gemini-3.1-flash-lite             alice-proposes= 70.0%  (n=1429)
    openai/qwen3.6-27b                       alice-proposes= 65.9%  (n=255)
    openai/qwen3.8-27b                       alice-proposes= 81.1%  (n=175)
    bedrock/converse/global.anthropic.claude-sonnet-4-6 alice-proposes= 74.1%  (n=135)
    bedrock/converse/global.anthropic.claude-opus-4-6-v1 alice-proposes= 65.2%  (n=69)
    deepseek/deepseek-v4-flash               alice-proposes= 84.6%  (n=26)
    openai/gpt-5.6-terra                     alice-proposes= 75.0%  (n=4)
    openai/gemma-4-31b-it                    alice-proposes=100.0%  (n=2)

    as Bob (model playing the Bob role -> does a stronger Bob make Alice propose less?):
    gemini/gemini-3.7-flash                  alice-proposes= 75.1%  (n=3669)
    gemini/gemini-3.1-flash-lite             alice-proposes= 75.7%  (n=630)
    openai/qwen3.8-27b                       alice-proposes= 74.8%  (n=306)
    bedrock/converse/global.anthropic.claude-sonnet-4-6 alice-proposes= 67.4%  (n=258)
    openai/qwen3.6-27b                       alice-proposes= 65.9%  (n=255)
    bedrock/converse/global.anthropic.claude-opus-4-6-v1 alice-proposes= 69.4%  (n=196)
    deepseek/deepseek-v4-flash               alice-proposes= 88.3%  (n=103)
    openai/gpt-5.6-luna                      alice-proposes= 80.0%  (n=15)
    openai/gpt-5.6-terra                     alice-proposes= 75.0%  (n=4)
    openai/gemma-4-31b-it                    alice-proposes=100.0%  (n=2)
