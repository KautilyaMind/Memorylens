# MemoryLens v1.0 Retrieval Evaluation

Measured at `2026-10-01T13:53:42.988237+00:00` using 2 steady-state run(s) per question/mode. Unanswerable questions are excluded from ranking aggregates.

## Aggregate results

| Method | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 | Median retrieval ms | p95 retrieval ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.417 | 0.531 | 0.683 | 0.450 | 0.415 | 0.470 | 27.2 | 40.2 |
| BM25 | 0.533 | 0.639 | 0.828 | 0.547 | 0.507 | 0.571 | 2.7 | 3.8 |
| Hybrid | 0.511 | 0.689 | 0.789 | 0.511 | 0.519 | 0.551 | 28.1 | 39.6 |
| Hybrid + Reranker | 0.583 | 0.733 | 0.864 | 0.593 | 0.582 | 0.632 | 1603.6 | 2181.7 |

## Quality versus latency

Compared with Hybrid, Hybrid + Reranker changed mean Recall@5 from 0.689 to 0.733, MRR from 0.511 to 0.593, and nDCG@5 from 0.519 to 0.582. Median retrieval latency changed from 28.1 ms to 1603.6 ms (57.0× on this hardware).

These are measurements for this frozen benchmark, not a general performance claim.

Per-question Recall@5 population standard deviations were Dense 0.439, BM25 0.437, Hybrid 0.409, and Hybrid + Reranker 0.395. The spread reinforces the category and failure-case analysis rather than relying only on means.

Metrics use positive relevance grades as relevant for Recall/MRR. nDCG uses gain `2^grade - 1` and logarithmic discount `log2(rank + 1)`.

## Category results

| Method | Category | Questions | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|
| Dense | comparison | 10 | 0.517 | 0.372 | 0.335 |
| Dense | cross_document | 10 | 0.467 | 0.603 | 0.426 |
| Dense | exact_terminology | 10 | 0.600 | 0.338 | 0.374 |
| Dense | numerical | 10 | 0.250 | 0.338 | 0.270 |
| Dense | semantic | 10 | 0.650 | 0.575 | 0.553 |
| Dense | table | 10 | 0.700 | 0.475 | 0.532 |
| BM25 | comparison | 10 | 0.633 | 0.633 | 0.551 |
| BM25 | cross_document | 10 | 0.350 | 0.380 | 0.291 |
| BM25 | exact_terminology | 10 | 0.900 | 0.550 | 0.626 |
| BM25 | numerical | 10 | 0.700 | 0.649 | 0.591 |
| BM25 | semantic | 10 | 0.750 | 0.650 | 0.596 |
| BM25 | table | 10 | 0.500 | 0.418 | 0.389 |
| Hybrid | comparison | 10 | 0.667 | 0.533 | 0.489 |
| Hybrid | cross_document | 10 | 0.517 | 0.514 | 0.426 |
| Hybrid | exact_terminology | 10 | 0.800 | 0.615 | 0.626 |
| Hybrid | numerical | 10 | 0.700 | 0.454 | 0.500 |
| Hybrid | semantic | 10 | 0.750 | 0.493 | 0.550 |
| Hybrid | table | 10 | 0.700 | 0.458 | 0.519 |
| Hybrid + Reranker | comparison | 10 | 0.617 | 0.558 | 0.504 |
| Hybrid + Reranker | cross_document | 10 | 0.333 | 0.421 | 0.321 |
| Hybrid + Reranker | exact_terminology | 10 | 0.800 | 0.745 | 0.720 |
| Hybrid + Reranker | numerical | 10 | 0.800 | 0.662 | 0.650 |
| Hybrid + Reranker | semantic | 10 | 0.850 | 0.683 | 0.686 |
| Hybrid + Reranker | table | 10 | 1.000 | 0.487 | 0.614 |

### Where simpler retrieval matched or beat advanced Recall@5

- cross_document: Dense 0.467 vs. Hybrid + Reranker 0.333
- comparison: BM25 0.633 vs. Hybrid + Reranker 0.617
- cross_document: BM25 0.350 vs. Hybrid + Reranker 0.333
- exact_terminology: BM25 0.900 vs. Hybrid + Reranker 0.800
- comparison: Hybrid 0.667 vs. Hybrid + Reranker 0.617
- cross_document: Hybrid 0.517 vs. Hybrid + Reranker 0.333
- exact_terminology: Hybrid 0.800 vs. Hybrid + Reranker 0.800

## Error analysis

107 question/method pairs did not retrieve all judged-relevant chunks in the top five. Representative cases:

### Q001 — dense

Why is high-bandwidth memory useful for AI accelerators?

Expected: `MICRON-DRAM-001_CH_0001, MICRON-TECH-002_CH_0028`
Retrieved top 5: `MICRON-TECH-001_CH_0014, MICRON-TECH-002_CH_0016, MICRON-TECH-002_CH_0015, MICRON-TECH-003_CH_0002, MICRON-TECH-001_CH_0006`

Expected passage: Introducing memory built for AI innovation  Micron® HBM3E Product highlights  • Advanced packaging with CoWoS, SiPWe are in the beginning of a golden era of artificial intelligence (AI),  support where AI is expected to be a central part of our everyday lives. This  • Better value than the competitiveproliferation is being fueled by advances in compute and memory  HBM in the market with:technologies.  - 50% higher capacity  High bandwidth memory (HBM) is at the forefront of these - 50% higher ba

Top retrieved passage: enables extremely high bandwidth by providing a very wide memory interface to the host processor (typically an ASIC, GPU, or FPGA). HBM memories also benefit from lower power consumption and a smaller footprint than 2D DRAM solutions, but these benefits come at a cost premium due to the unique manufacturing process and additional silicon required.  The latest generation of HBM (HBM2E) features increased bandwidth of up to 410 GB/s and up to 16GB density.  It is likely that AI applications with t

### Q001 — bm25

Why is high-bandwidth memory useful for AI accelerators?

Expected: `MICRON-DRAM-001_CH_0001, MICRON-TECH-002_CH_0028`
Retrieved top 5: `MICRON-TECH-002_CH_0031, MICRON-TECH-002_CH_0003, MICRON-TECH-002_CH_0017, MICRON-TECH-002_CH_0007, MICRON-DRAM-003_CH_0002`

Expected passage: Introducing memory built for AI innovation  Micron® HBM3E Product highlights  • Advanced packaging with CoWoS, SiPWe are in the beginning of a golden era of artificial intelligence (AI),  support where AI is expected to be a central part of our everyday lives. This  • Better value than the competitiveproliferation is being fueled by advances in compute and memory  HBM in the market with:technologies.  - 50% higher capacity  High bandwidth memory (HBM) is at the forefront of these - 50% higher ba

Top retrieved passage: Micron and the High-Bandwidth Memory market face several challenges despite its critical role in high-performance computing, artificial intelligence, and data-intensive applications. Some of these challenges are:  • High manufacturing costs: HBM is more expensive to produce than traditional memory  technologies because of its complex 3D stacking architecture and the need for advanced packaging techniques such as through-silicon vias. This raises the overall cost of systems that use HBM, limiting

### Q001 — hybrid

Why is high-bandwidth memory useful for AI accelerators?

Expected: `MICRON-DRAM-001_CH_0001, MICRON-TECH-002_CH_0028`
Retrieved top 5: `MICRON-TECH-003_CH_0002, MICRON-TECH-002_CH_0017, MICRON-TECH-002_CH_0015, MICRON-TECH-001_CH_0006, MICRON-TECH-003_CH_0022`

Expected passage: Introducing memory built for AI innovation  Micron® HBM3E Product highlights  • Advanced packaging with CoWoS, SiPWe are in the beginning of a golden era of artificial intelligence (AI),  support where AI is expected to be a central part of our everyday lives. This  • Better value than the competitiveproliferation is being fueled by advances in compute and memory  HBM in the market with:technologies.  - 50% higher capacity  High bandwidth memory (HBM) is at the forefront of these - 50% higher ba

Top retrieved passage: tion in memory and computing technologies, especially around deploying AI-specific infrastructures and reshaping the architecture and capabilities of modern data centers. All data centers will become AI data centers to some extent.  This white paper explores the impact of new technologies on memory providers and asserts that advanced technologies such as high bandwidth memory (HBM) are fundamentally altering the industry landscape. Instead of being relegated to lowermargin commodity status, memo

### Q001 — hybrid_rerank

Why is high-bandwidth memory useful for AI accelerators?

Expected: `MICRON-DRAM-001_CH_0001, MICRON-TECH-002_CH_0028`
Retrieved top 5: `MICRON-TECH-002_CH_0016, MICRON-TECH-001_CH_0014, MICRON-TECH-002_CH_0031, MICRON-TECH-002_CH_0017, MICRON-TECH-001_CH_0006`

Expected passage: Introducing memory built for AI innovation  Micron® HBM3E Product highlights  • Advanced packaging with CoWoS, SiPWe are in the beginning of a golden era of artificial intelligence (AI),  support where AI is expected to be a central part of our everyday lives. This  • Better value than the competitiveproliferation is being fueled by advances in compute and memory  HBM in the market with:technologies.  - 50% higher capacity  High bandwidth memory (HBM) is at the forefront of these - 50% higher ba

Top retrieved passage: efficiency of AI and data-intensive workloads, acting as a key enabler for advanced applications. AI models require large-scale data processing, and the performance of these workloads is directly linked to memory attributes such as bandwidth, capacity, and latency. High memory bandwidth allows for faster data movement between memory and processing units, boosting computational speed, while higher memory capacity ensures that vast data sets can be stored and processed without CPU offloading and d

### Q002 — dense

What role does mobile LPDRAM play in edge AI inference?

Expected: `MICRON-DRAM-003_CH_0003, MICRON-DRAM-003_CH_0009`
Retrieved top 5: `MICRON-DRAM-003_CH_0003, MICRON-DRAM-003_CH_0001, MICRON-DRAM-003_CH_0019, MICRON-DRAM-003_CH_0002, MICRON-DRAM-003_CH_0020`

Expected passage: he transition to edge AI is driven by three key enablers:  • Small language models (SLMs)  • High-performance edge AI accelerators  • Advanced memory technologies like mobile LPDRAM  As edge AI capabilities expand, they increasingly support agentic AI systems that place even greater demands on memory performance, especially during the decode phase of inference. This paper explores how memory, particularly mobile LPDRAM, underpins the performance of edge AI workloads. We examine the inference pip

Top retrieved passage: he transition to edge AI is driven by three key enablers:  • Small language models (SLMs)  • High-performance edge AI accelerators  • Advanced memory technologies like mobile LPDRAM  As edge AI capabilities expand, they increasingly support agentic AI systems that place even greater demands on memory performance, especially during the decode phase of inference. This paper explores how memory, particularly mobile LPDRAM, underpins the performance of edge AI workloads. We examine the inference pip

### Q002 — hybrid

What role does mobile LPDRAM play in edge AI inference?

Expected: `MICRON-DRAM-003_CH_0003, MICRON-DRAM-003_CH_0009`
Retrieved top 5: `MICRON-DRAM-003_CH_0001, MICRON-DRAM-003_CH_0003, MICRON-DRAM-003_CH_0019, MICRON-DRAM-003_CH_0002, MICRON-TECH-001_CH_0008`

Expected passage: he transition to edge AI is driven by three key enablers:  • Small language models (SLMs)  • High-performance edge AI accelerators  • Advanced memory technologies like mobile LPDRAM  As edge AI capabilities expand, they increasingly support agentic AI systems that place even greater demands on memory performance, especially during the decode phase of inference. This paper explores how memory, particularly mobile LPDRAM, underpins the performance of edge AI workloads. We examine the inference pip

Top retrieved passage: Decode at the edge: the role of LPDRAM in accelerating AI inference  Executive summary  Edge AI is now a core driver of innovation, delivering low-latency intelligence directly on smartphones and laptops. These edge devices are now capable of running intelligent, context-aware workloads locally, without relying on the cloud. However, as AI shifts from cloud to edge, memory becomes a critical bottleneck — placing it at the center of system performance. From centralized AI models to distributed, a

### Q004 — dense

What architectural objective drove the DDR5 redesign?

Expected: `MICRON-DRAM-005_CH_0004, MICRON-DRAM-005_CH_0002`
Retrieved top 5: `MICRON-TECH-001_CH_0012, MICRON-DRAM-005_CH_0004, MICRON-DRAM-005_CH_0001, MICRON-DRAM-007_CH_0018, MICRON-TECH-001_CH_0011`

Expected passage: DDR5 Features  The transition from DDR4 to DDR5 represents far more than a typical DDR SDRAM generational change. DDR5 demonstrates a major step forward that has completely overhauled the overall DDR architecture with one primary goal: increasing bandwidth.  Increased Data Rates  A number of key feature additions and improvements enable DDR5’s bandwidth increase. Primary among these is a dramatic increase in device data rates. While DDR4 spanned data rates from 1600 MT/s to 3200 MT/s, DDR5 is cu

Top retrieved passage: Similar to the transition to DDR5, the LPDDR4 to LPDDR5 transition is also set to boost memory bandwidth, while reducing runtime power use and utilizing features to reduce overall power consumption. Where DDR5 will be found mainly in datacenters and PCs, LPDDR5 will target edge AI applications and mobile devices where size, cost, and power are the most significant constraints. Performance and power advantages may drive LPDDR5 beyond traditional industries and applications.

### Q004 — bm25

What architectural objective drove the DDR5 redesign?

Expected: `MICRON-DRAM-005_CH_0004, MICRON-DRAM-005_CH_0002`
Retrieved top 5: `MICRON-DRAM-005_CH_0002, MICRON-DRAM-007_CH_0008, MICRON-DRAM-007_CH_0013, MICRON-DRAM-007_CH_0017, MICRON-NOR-002_CH_0001`

Expected passage: DDR5 Features  The transition from DDR4 to DDR5 represents far more than a typical DDR SDRAM generational change. DDR5 demonstrates a major step forward that has completely overhauled the overall DDR architecture with one primary goal: increasing bandwidth.  Increased Data Rates  A number of key feature additions and improvements enable DDR5’s bandwidth increase. Primary among these is a dramatic increase in device data rates. While DDR4 spanned data rates from 1600 MT/s to 3200 MT/s, DDR5 is cu

Top retrieved passage: evel simulation example indicates an approximate performance increase of 1.36X effective bandwidth. At a higher data rate, DDR5-4800, the approximate performance increase becomes 1.87X—nearly double the bandwidth as compared to DDR4-3200.  Figure 1: Effective Bandwidth: DDR4 vs. DDR51  Driven by data rates up to 6400 MT/s and key architectural improvements, Micron’s DDR5 is pushing potential system bandwidth even higher. This white paper discusses some of the key architectural improvements of DD

### Q005 — hybrid_rerank

How does same-bank refresh reduce DDR5 disruption?

Expected: `MICRON-DRAM-006_CH_0005, MICRON-DRAM-006_CH_0007`
Retrieved top 5: `MICRON-DRAM-006_CH_0006, MICRON-DRAM-006_CH_0007, MICRON-DRAM-005_CH_0010, MICRON-DRAM-006_CH_0008, MICRON-DRAM-006_CH_0004`

Expected passage: interval (timing parameter tREFI). For REFab commands, the system must ensure all banks are idle prior to issuing the command, on an average of once every 3.9µs in "normal" refresh mode, with a duration of 295ns for a 16Gb DDR5 SDRAM device.  The performance benefit of the REFsb command is that only one bank in each bank group needs to be idle before issuing the command. The remaining 12 banks (for a 16Gb, x4/x8 device; blue cells in Figure 3) do not have to be idle when the REFsb command is iss

Top retrieved passage: ich also reduces the system access lockout (tRFCsb) to actively refreshing banks (red cells in Figure 3). A restriction when using REFsb is that each "same bank" must receive one REFsb command prior to that "same bank" being issued a second REFsb command, but the REFsb commands can be issued in any bank order.

### Q007 — dense

How do the typical roles of NOR flash and NAND flash differ?

Expected: `MICRON-NOR-001_CH_0002, MICRON-NOR-001_CH_0005`
Retrieved top 5: `MICRON-NOR-001_TABLE_0001, MICRON-NOR-001_TABLE_0002, MICRON-NOR-001_TABLE_0003, MICRON-NOR-001_CH_0005, MICRON-NOR-001_CH_0014`

Expected passage: olution.  Getting to know NOR Flash NOR ﬂash devices, available in densities from 128Mb up infotainment), IPC/factory automation, intelligent edge to 2Gb, are primarily used for reliable code storage (boot, devices, machine to machine, healthcare, radio access application, OS, and execute-in-place [XIP] code in an networks (RAN), servers and routers, wearables, cameras, embedded system) and frequently changing small data and printers. storage. NOR ﬂash provides systems with the fastest bootable

Top retrieved passage: Document: Micron NOR and NAND Flash Guide Table: NAND Controller Page: 2 Columns: NOR and NAND features comparison | Column 2 NOR and NAND features comparison: Xccela Octal Flash, Serial NOR, Parallel NOR • Lower density • Low pin count (Octal and Serial) • Ease-of-use • Reliable code and data storage • Fast read and random access times • Higher endurance and data retention | Column 2: SLC/MLC/TLC/SPI NAND, Managed NAND • Higher density, low pin count • Requires controller management (SLC, MLC)

### Q007 — bm25

How do the typical roles of NOR flash and NAND flash differ?

Expected: `MICRON-NOR-001_CH_0002, MICRON-NOR-001_CH_0005`
Retrieved top 5: `MICRON-NOR-001_CH_0004, MICRON-NOR-001_TABLE_0001, MICRON-NOR-001_CH_0008, MICRON-NOR-001_TABLE_0002, MICRON-NOR-001_CH_0027`

Expected passage: olution.  Getting to know NOR Flash NOR ﬂash devices, available in densities from 128Mb up infotainment), IPC/factory automation, intelligent edge to 2Gb, are primarily used for reliable code storage (boot, devices, machine to machine, healthcare, radio access application, OS, and execute-in-place [XIP] code in an networks (RAN), servers and routers, wearables, cameras, embedded system) and frequently changing small data and printers. storage. NOR ﬂash provides systems with the fastest bootable

Top retrieved passage: (10x13mm) µC µC (6x8mm)  Parallel NOR Flash (50 active pins) Xccela™ Octal Flash (11 active pins)  NOR | NAND Flash guide

### Q007 — hybrid

How do the typical roles of NOR flash and NAND flash differ?

Expected: `MICRON-NOR-001_CH_0002, MICRON-NOR-001_CH_0005`
Retrieved top 5: `MICRON-NOR-001_TABLE_0001, MICRON-NOR-001_TABLE_0002, MICRON-NOR-001_CH_0004, MICRON-NOR-001_CH_0008, MICRON-NOR-001_CH_0027`

Expected passage: olution.  Getting to know NOR Flash NOR ﬂash devices, available in densities from 128Mb up infotainment), IPC/factory automation, intelligent edge to 2Gb, are primarily used for reliable code storage (boot, devices, machine to machine, healthcare, radio access application, OS, and execute-in-place [XIP] code in an networks (RAN), servers and routers, wearables, cameras, embedded system) and frequently changing small data and printers. storage. NOR ﬂash provides systems with the fastest bootable

Top retrieved passage: Document: Micron NOR and NAND Flash Guide Table: NAND Controller Page: 2 Columns: NOR and NAND features comparison | Column 2 NOR and NAND features comparison: Xccela Octal Flash, Serial NOR, Parallel NOR • Lower density • Low pin count (Octal and Serial) • Ease-of-use • Reliable code and data storage • Fast read and random access times • Higher endurance and data retention | Column 2: SLC/MLC/TLC/SPI NAND, Managed NAND • Higher density, low pin count • Requires controller management (SLC, MLC)

## Limitations

- Relevance judgments are corpus-grounded but require independent human sign-off before being described as a final human-reviewed benchmark.
- A valid citation marker does not establish that the cited passage entails a generated claim.
- Results apply only to the frozen corpus, models, chunking configuration, and hardware recorded with this run.
- The benchmark measures retrieval ranking; answer correctness and faithfulness use a separate manual-review workflow.
