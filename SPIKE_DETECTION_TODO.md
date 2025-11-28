# Spike Detection Processing Reference

This document lists all EEG files to process, the channels to analyze, and seizure times for filtering.

## Processing Instructions

1. Run headless spike detection for each file listed below
2. Use the **Recommended Channel** specified for each mouse
3. Update `SpikeDetection.py` line 194 with the seizure times before running (if seizures present)
4. Times are in **seconds from 10:00 AM** (where 10:00 = 0 seconds)

---

## Dataset #1: Mice 4244, 4245, 4252, 4248, 4254, 4255, 4256

### 1. 2025-05-07 (Baseline - No injection)
**Status:** Not exported yet

**Mice and Channels:**
- **Mouse 4244** - Channel: `4 RP` ✓ No seizures
- **Mouse 4245** - Channel: `1 RP` ✓ No seizures
- **Mouse 4252** - Channel: `C14` ✓ No seizures
- **Mouse 4248** - Channel: `C110` ✓ No seizures
- **Mouse 4254** - Channel: `C102` ✓ No seizures
- **Mouse 4255** - Channel: `7 RP` ✓ No seizures
- **Mouse 4256** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(21420.0, 21447.4)]
  ```

---

### 2. 2025-05-08 (Baseline - No injection)
**Status:** Not exported yet

**Mice and Channels:**
- **Mouse 4244** - Channel: `4 RP` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(11460.0, 11589.8)]
  ```
- **Mouse 4245** - Channel: `1 RP` ✓ No seizures
- **Mouse 4252** - Channel: `C14` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(20400.0, 20440.3)]
  ```
- **Mouse 4248** - Channel: `C110` ✓ No seizures
- **Mouse 4254** - Channel: `C102` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(7920.0, 7975.0)]
  ```
- **Mouse 4255** - Channel: `7 RP` ✓ No seizures
- **Mouse 4256** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(15720.0, 15745.5)]
  ```

---

### 3. xiaogang~ zhan_02520025-e690-46cb-a73c-8bfad7d93203.EDF
**Date:** 2025-05-09
**Recording Start:** 09:01:54 → **0 seconds = 12:01:54**
**Injection:** CNO around 10am (09:01:54)

**Mice and Channels:**
- **Mouse 4244** - Channel: `4 RP` ✓ No seizures
- **Mouse 4245** - Channel: `1 RP` ✓ No seizures
- **Mouse 4252** - Channel: `C14` ✓ No seizures
- **Mouse 4248** - Channel: `C110` ✓ No seizures
- **Mouse 4254** - Channel: `C102` ✓ No seizures
- **Mouse 4255** - Channel: `7 RP` ✓ No seizures
- **Mouse 4256** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(13266.0, 13295.1)]
  ```

---

### 4. xiaogang~ zhan_58b1b97e-0f8c-4b5d-bfba-fac0c42a0735.EDF
**Date:** 2025-05-10
**Recording Start:** 07:00:12 → **0 seconds = 10:00:12**
**Injection:** Vehicle around 10am (07:00:12)

**Mice and Channels:**
- **Mouse 4244** - Channel: `4 RP` ✓ No seizures
- **Mouse 4245** - Channel: `1 RP` ✓ No seizures
- **Mouse 4252** - Channel: `C14` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(13008.0, 13043.5)]
  ```
- **Mouse 4248** - Channel: `C110` ✓ No seizures
- **Mouse 4254** - Channel: `C102` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(2388.0, 2475.6)]
  ```
- **Mouse 4255** - Channel: `7 RP` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(15168.0, 15195.1)]
  ```
- **Mouse 4256** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(18288.0, 18317.3)]
  ```

---

### 5. xiaogang~ zhan_15e0da1a-f7ca-4555-9493-43aaefe43635.EDF
**Date:** 2025-05-11
**Recording Start:** 07:00:12 → **0 seconds = 10:00:12**
**Injection:** CNO or Vehicle around 10am (07:00:12)

**Mice and Channels:**
- **Mouse 4244** - Channel: `4 RP` ✓ No seizures
- **Mouse 4245** - Channel: `1 RP` ✓ No seizures
- **Mouse 4252** - Channel: `C14` ✓ No seizures
- **Mouse 4248** - Channel: `C110` ✓ No seizures
- **Mouse 4254** - Channel: `C102` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(21408.0, 21468.9)]
  ```
- **Mouse 4255** - Channel: `7 RP` ✓ No seizures
- **Mouse 4256** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(3648.0, 3691.1), (13968.0, 13997.2)]
  ```

---

### 6. xiaogang~ zhan_32b9a988-8e21-440a-a93f-60da126d9484.EDF
**Date:** 2025-05-12
**Recording Start:** 07:00:12 → **0 seconds = 10:00:12**
**Injection:** CNO or Vehicle around 10am (07:00:12)

**Mice and Channels:**
- **Mouse 4244** - Channel: `4 RP` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(13308.0, 13334.0)]
  ```
- **Mouse 4245** - Channel: `1 RP` ✓ No seizures
- **Mouse 4252** - Channel: `C14` ✓ No seizures
- **Mouse 4248** - Channel: `C110` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(19368.0, 19436.9)]
  ```
- **Mouse 4254** - Channel: `C102` ✓ No seizures
- **Mouse 4255** - Channel: `7 RP` ✓ No seizures
- **Mouse 4256** - Channel: `C94` ✓ No seizures

---

### 7. xiaogang~ zhan_f60d5400-4f2e-47be-b3f4-7eb667cb2bd8.EDF
**Date:** 2025-05-13
**Recording Start:** 07:00:12 → **0 seconds = 10:00:12**
**Injection:** CNO or Vehicle around 10am (07:00:12)

**Mice and Channels:**
- **Mouse 4244** - Channel: `4 RP` ✓ No seizures
- **Mouse 4245** - Channel: `1 RP` ✓ No seizures
- **Mouse 4252** - Channel: `C14` ✓ No seizures
- **Mouse 4248** - Channel: `C110` ✓ No seizures
- **Mouse 4254** - Channel: `C102` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(11808.0, 11861.6), (20028.0, 20112.9)]
  ```
- **Mouse 4255** - Channel: `7 RP` ✓ No seizures
- **Mouse 4256** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(10608.0, 10638.5), (15888.0, 15922.3)]
  ```

---

### 8. xiaogang~ zhan_5b7d5ec6-3284-48bf-b680-8786ce8b1060.EDF
**Date:** 2025-05-14
**Recording Start:** 07:00:13 → **0 seconds = 10:00:13**
**Injection:** CNO or Vehicle around 10am (07:00:13)

**Mice and Channels:**
- **Mouse 4244** - Channel: `4 RP` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(2327.0, 2360.7), (5147.0, 5181.8)]
  ```
- **Mouse 4245** - Channel: `1 RP` ✓ No seizures
- **Mouse 4252** - Channel: `C14` ✓ No seizures
- **Mouse 4248** - Channel: `C110` ✓ No seizures
- **Mouse 4254** - Channel: `C102` ✓ No seizures
- **Mouse 4255** - Channel: `7 RP` ✓ No seizures
- **Mouse 4256** - Channel: `C94` ✓ No seizures

---

## Dataset #2: Mice 4759, 4763, 4764, 4765

### 9. p35~ p60induce_3f55396b-8cb5-4b7f-8c28-65898c40bc1e.EDF
**Date:** 2025-09-02
**Recording Start:** 07:00:00 (default) → **0 seconds = 10:00:00**
**Injection:** None (baseline)

**Mice and Channels:**
- **Mouse 4759** - Channel: `9 RP` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(16560.0, 16601.2)]
  ```
- **Mouse 4763** - Channel: `5 RP` ✓ No seizures
- **Mouse 4764** - Channel: `4 RP` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(15540.0, 15631.0), (15660.0, 15697.7)]
  ```
- **Mouse 4765** - Channel: `C94` ✓ No seizures

---

### 10. p35~ p60induce_ab30705f-6fd1-43c9-9c8b-e7845c325b39.EDF
**Date:** 2025-09-03
**Recording Start:** 07:00:10 → **0 seconds = 10:00:10**
**Injection:** CNO or Vehicle around 10am (07:00:10)

**Mice and Channels:**
- **Mouse 4759** - Channel: `9 RP` ✓ No seizures
- **Mouse 4763** - Channel: `5 RP` ✓ No seizures
- **Mouse 4764** - Channel: `4 RP` ✓ No seizures
- **Mouse 4765** - Channel: `C94` ✓ No seizures

---

### 11. p35~ p60induce_23ac7e5f-837e-4b75-b6a9-9bbf497c263d.EDF
**Date:** 2025-09-04
**Recording Start:** 07:00:10 → **0 seconds = 10:00:10**
**Injection:** CNO or Vehicle around 10am (07:00:10)

**Mice and Channels:**
- **Mouse 4759** - Channel: `9 RP` ✓ No seizures
- **Mouse 4763** - Channel: `5 RP` ✓ No seizures
- **Mouse 4764** - Channel: `4 RP` ✓ No seizures
- **Mouse 4765** - Channel: `C94` ✓ No seizures

---

### 12. p35~ p60induce_78143b16-2a60-4524-9a4f-57bf0acdbd73.EDF
**Date:** 2025-09-05
**Recording Start:** 07:00:11 → **0 seconds = 10:00:11**
**Injection:** CNO or Vehicle around 10am (07:00:11)

**Mice and Channels:**
- **Mouse 4759** - Channel: `9 RP` ✓ No seizures
- **Mouse 4763** - Channel: `5 RP` ✓ No seizures
- **Mouse 4764** - Channel: `4 RP` ✓ No seizures
- **Mouse 4765** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(7429.0, 7482.9)]
  ```

---

### 13. p35~ p60induce_284ffe4f-3c88-40be-a3ab-0e03accaf127.EDF
**Date:** 2025-09-06
**Recording Start:** 07:00:11 → **0 seconds = 10:00:11**
**Injection:** CNO or Vehicle around 10am (07:00:11)

**Mice and Channels:**
- **Mouse 4759** - Channel: `9 RP` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(21529.0, 21546.2)]
  ```
- **Mouse 4763** - Channel: `5 RP` ✓ No seizures
- **Mouse 4764** - Channel: `4 RP` ✓ No seizures
- **Mouse 4765** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(13189.0, 13235.2), (13429.0, 13504.4)]
  ```

---

### 14. p35~ p60induce_0003c5e0-8057-4fbc-8da2-fe18d4649458.EDF
**Date:** 2025-09-07
**Recording Start:** 07:00:10 → **0 seconds = 10:00:10**
**Injection:** CNO or Vehicle around 10am (07:00:10)

**Mice and Channels:**
- **Mouse 4759** - Channel: `9 RP` ✓ No seizures
- **Mouse 4763** - Channel: `5 RP` ✓ No seizures
- **Mouse 4764** - Channel: `4 RP` ✓ No seizures
- **Mouse 4765** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(4790.0, 4869.0), (16250.0, 16323.7)]
  ```

---

### 15. p35~ p60induce_8f7c70af-dc9c-43f0-8206-004d1979408a.EDF
**Date:** 2025-09-08
**Recording Start:** 07:00:10 → **0 seconds = 10:00:10**
**Injection:** CNO or Vehicle around 10am (07:00:10)

**Mice and Channels:**
- **Mouse 4759** - Channel: `9 RP` ✓ No seizures
- **Mouse 4763** - Channel: `5 RP` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(7190.0, 7219.1)]
  ```
- **Mouse 4764** - Channel: `4 RP` ✓ No seizures
- **Mouse 4765** - Channel: `C94` ⚠️ **HAS SEIZURES**
  ```python
  seizureTimes = [(3110.0, 3197.5), (19730.0, 19791.0)]
  ```

---

## Summary Statistics

**Total Files:** 15 (2 not exported yet)
**Total Mice:** 11 unique mice
**Total Processing Runs:** 13 files × average 4-7 mice each = ~80 channel analyses
**Files with Seizures:** 13 out of 15

---

## Channel Information by Mouse

| Mouse ID | Available Channels | Recommended Channel |
|----------|-------------------|---------------------|
| 4244 | 4RP, 4LP | `4 RP` |
| 4245 | 1RP, 1LP | `1 RP` |
| 4248 | C110, C111 | `C110` |
| 4252 | C14, C15 | `C14` |
| 4254 | C102, C103 | `C102` |
| 4255 | 7RP, 7LP | `7 RP` |
| 4256 | C94, C95 | `C94` |
| 4759 | 9RP, 9LP | `9 RP` |
| 4763 | 5RP, 5LP | `5 RP` |
| 4764 | 4RP, 4LP | `4 RP` |
| 4765 | C94, C95 | `C94` |

---

## Running the Analysis

### Example Command

```bash
python headless_spike_detection.py \
    --eeg-file "xiaogang~ zhan_02520025-e690-46cb-a73c-8bfad7d93203.EDF" \
    --channel "4 RP" \
    --output-dir "./output"
```

### Before Each Run

1. Check this document for the mouse/date combination you're processing
2. If seizures are present, update `SpikeDetection.py` line 194:
   ```python
   seizureTimes = [(start, end), ...]  # Copy from this document
   ```
3. If no seizures, use:
   ```python
   seizureTimes = []
   ```

---

## Notes

- All seizure times are in **seconds from 10:00 AM** (10:00 = 0, 16:00 = 21600)
- Channel names with spaces need quotes: `--channel "4 RP"`
- Files marked "Not exported yet" may not be available for processing
