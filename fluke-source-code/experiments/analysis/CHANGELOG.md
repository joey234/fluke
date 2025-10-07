# FLUKE Analysis Framework - Changelog

## Version 1.1 - New Model Integration

### Added Support for New Frontier Models

#### Models Added:
- **GPT-5** (gpt-5-standard): OpenAI's latest model
- **DeepSeek R1** (deepseek-r1-deepseek): DeepSeek's reasoning model

### Files Updated:

1. **utils.py**
   - Added GPT-5 and DeepSeek R1 to `Config.LLM_MODELS`
   - Updated `normalize_model_name()` to handle new model naming patterns
   - Added mappings for various naming conventions of new models

2. **consolidated_analysis.py**
   - Updated model list to include new frontier models
   - Enhanced `_parse_llm_filename()` to correctly parse GPT-5 and DeepSeek R1 result files
   - Added special handling for model-specific filename patterns

3. **visualization.py**
   - Added distinctive colors for new models:
     - GPT-5: Bright red (#ff6b6b)
     - DeepSeek R1: Orange (#ff9f43)
   - Ensures visual distinction in comparison charts

4. **run_analysis.py**
   - Added frontier model analysis section in reports
   - Automatic detection and highlighting of GPT-5 and DeepSeek R1 results
   - Performance comparison between GPT-5 and GPT-4o
   - Task-wise breakdown for new models

5. **README.md**
   - Updated supported models section
   - Added GPT-5 and DeepSeek R1 to LLM list
   - Documented new model support as an improvement

### Features:

- **Automatic Detection**: Framework automatically detects and processes GPT-5 and DeepSeek R1 results
- **Backward Compatible**: All existing functionality preserved
- **Enhanced Reporting**: Special sections for frontier model analysis
- **Visual Distinction**: New models have unique colors in visualizations

### Usage:

```bash
# Analyze all tasks including new models
python run_analysis.py --task all --visualize --report

# The framework will automatically:
# - Detect GPT-5 and DeepSeek R1 results
# - Include them in all analyses
# - Generate frontier model comparisons
# - Highlight performance differences
```

### Result File Patterns Supported:

- GPT-5: `gpt-5-standard-0shot-{modification}.csv`
- DeepSeek R1: `deepseek-r1-deepseek-0shot-{modification}.csv`

### Verification:

The integration was tested and verified to correctly:
- Load GPT-5 results (found in coref, dialogue, ner, sa tasks)
- Load DeepSeek R1 results (found in coref task)
- Parse filenames correctly
- Generate accurate reports
- Create proper visualizations

### Statistics from Test Run:

- **COREF**: 18 GPT-5 entries, 11 DeepSeek R1 entries
- **DIALOGUE**: 18 GPT-5 entries
- **NER**: 18 GPT-5 entries
- **SA**: 19 GPT-5 entries

Total new model experiments integrated: 84+ entries