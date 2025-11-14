# DESTINY Symbolic Modeling - File Index

Quick reference guide to all symbolic modeling files.

## 📖 Start Here

**New to symbolic modeling?** → Read in this order:

1. **`SYMBOLIC_MODELING_COMPLETE.md`** - Complete overview and summary
2. **`QUICK_START_SYMBOLIC.md`** - Quick start guide
3. Run **`python example_symbolic_destiny.py`** - See it in action
4. **`SYMBOLIC_MODELING_README.md`** - Full documentation

## 🔧 Core Implementation Files

### `base_parameters.py` (410 lines) ⭐ MAIN FILE
**What**: Core symbolic parameters class
**Contains**:
- 87 symbolic variables (vdd, vth, capIdealGate, resistanceOn, etc.)
- tech_values dictionary for concrete values
- symbol_table for string → symbol mapping
- populate_from_* methods to extract values from DESTINY objects
**Use**: Import and use `BaseParameters()` class

### `symbolic_wrapper.py` (optional)
**What**: Optional utilities for parallel symbolic/numerical computation
**Note**: NOT the main approach - kept for reference only
**Use**: Can be ignored - base_parameters.py is the main approach

## 🧪 Test & Example Files

### `test_base_params.py`
**What**: Basic tests for symbolic parameters
**Tests**:
- Symbolic variable creation
- Concrete value population
- Symbolic evaluation
**Run**: `python test_base_params.py`

### `example_symbolic_destiny.py` ⭐ BEST EXAMPLES
**What**: 6 comprehensive examples with real circuit calculations
**Examples**:
1. Gate Capacitance
2. Transistor Resistance
3. Dynamic Power (with voltage scaling)
4. RC Delay (with technology scaling)
5. Memory Cell Write Energy
6. Complete CMOS Inverter Analysis
**Run**: `python example_symbolic_destiny.py`

### `comparison_demo.py`
**What**: Side-by-side CACTI vs DESTINY comparison
**Shows**: Both frameworks use exact same approach
**Run**: `python comparison_demo.py`

### `test_symbolic.py`
**What**: Tests for symbolic_wrapper.py utilities
**Note**: Optional - tests the wrapper approach (not main approach)

## 📚 Documentation Files

### `SYMBOLIC_MODELING_COMPLETE.md` ⭐ START HERE
**What**: Complete overview and implementation summary
**Contains**:
- What was implemented
- CACTI comparison
- Complete workflow example
- File descriptions
- Test results
- Integration instructions
- Use cases
**Audience**: Everyone - best overall summary

### `QUICK_START_SYMBOLIC.md`
**What**: Quick reference for getting started
**Contains**:
- How to run examples
- Basic usage pattern
- Available examples
- Key features demonstrated
- Tips and references
**Audience**: Quick start guide

### `SYMBOLIC_MODELING_README.md`
**What**: Full technical documentation
**Contains**:
- Complete list of 87 symbolic variables
- Detailed usage instructions
- Integration guide
- Example calculations
- Benefits and notes
**Audience**: Developers integrating symbolic modeling

### `SYMBOLIC_WALKTHROUGH.md`
**What**: Step-by-step comparison with CACTI
**Contains**:
- 5-step comparison showing identical approach
- Complete example comparison
- Key similarities table
- Detailed workflow comparison
- Advanced features
**Audience**: Those wanting to understand CACTI similarity

### `SYMBOLIC_INDEX.md` (this file)
**What**: Navigation guide for all symbolic modeling files
**Audience**: You! :)

## 📊 Quick Reference

### File Organization

```
destiny_3d_cache_python/
├── Core Implementation
│   ├── base_parameters.py          ⭐ Main symbolic parameters class
│   └── symbolic_wrapper.py          Optional utilities
│
├── Tests & Examples
│   ├── test_base_params.py          Basic tests
│   ├── example_symbolic_destiny.py  ⭐ Best examples (6 comprehensive)
│   ├── comparison_demo.py           CACTI vs DESTINY demo
│   └── test_symbolic.py             Optional wrapper tests
│
└── Documentation
    ├── SYMBOLIC_MODELING_COMPLETE.md    ⭐ Start here - complete overview
    ├── QUICK_START_SYMBOLIC.md          Quick start guide
    ├── SYMBOLIC_MODELING_README.md      Full technical docs
    ├── SYMBOLIC_WALKTHROUGH.md          CACTI comparison
    └── SYMBOLIC_INDEX.md                This navigation guide
```

### What to Read Based on Your Goal

**Goal: Quick overview** → `SYMBOLIC_MODELING_COMPLETE.md`

**Goal: Get started immediately** → `QUICK_START_SYMBOLIC.md` + run `example_symbolic_destiny.py`

**Goal: Understand all parameters** → `SYMBOLIC_MODELING_README.md`

**Goal: Compare with CACTI** → `SYMBOLIC_WALKTHROUGH.md` + run `comparison_demo.py`

**Goal: Integrate into code** → `SYMBOLIC_MODELING_README.md` (Integration section)

**Goal: See working examples** → Run `example_symbolic_destiny.py`

**Goal: Write new calculations** → Look at `example_symbolic_destiny.py` + `base_parameters.py`

### Quick Commands

```bash
# See all examples in action
python example_symbolic_destiny.py

# Test basic functionality
python test_base_params.py

# See CACTI comparison
python comparison_demo.py

# See list of all symbolic variables
python -c "from base_parameters import BaseParameters; bp = BaseParameters(); print(f'{len(bp.symbol_table)} variables:', ', '.join(sorted(bp.symbol_table.keys())[:10]), '...')"
```

## 🎯 Common Tasks

### I want to use symbolic parameters in my code

1. Import and create:
   ```python
   from base_parameters import BaseParameters
   bp = BaseParameters()
   ```

2. Populate values:
   ```python
   import globals as g
   from Technology import Technology

   g.tech = Technology()
   g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())
   bp.populate_from_technology(g.tech)
   ```

3. Use in expressions:
   ```python
   resistance = bp.vdd / bp.currentOnNmos
   result = resistance.evalf(subs=bp.tech_values)
   ```

See `example_symbolic_destiny.py` for complete working examples.

### I want to understand what variables are available

See `SYMBOLIC_MODELING_README.md` section "Symbolic Variables" for complete list of all 87 variables organized by category:
- Technology Parameters (25 variables)
- Memory Cell Parameters (28 variables)
- Wire Parameters (16 variables)
- TSV Parameters (10 variables)
- Input Parameters (8 variables)

### I want to verify this matches CACTI

1. Read `SYMBOLIC_WALKTHROUGH.md` - detailed step-by-step comparison
2. Run `python comparison_demo.py` - see side-by-side demonstration
3. Conclusion: **EXACT SAME APPROACH** ✓

### I want to add more symbolic variables

Edit `base_parameters.py`:

1. Add symbol creation in `__init__`:
   ```python
   self.my_new_param = symbols("my_new_param", positive=True, real=True)
   ```

2. Add to symbol_table in `build_symbol_table()`:
   ```python
   'my_new_param': self.my_new_param,
   ```

3. Add value population in appropriate `populate_from_*` method:
   ```python
   self.tech_values[self.my_new_param] = source.my_new_param
   ```

## 📈 Status

✅ **Implementation**: Complete
✅ **Testing**: All tests passing
✅ **Documentation**: Complete
✅ **CACTI Comparison**: Verified identical
✅ **Examples**: 6 comprehensive examples working

**Ready to use!**

## 🔗 Related Files (in main DESTINY)

These DESTINY files are used by symbolic modeling:

- `Technology.py` - Source of technology parameters
- `MemCell.py` - Source of memory cell parameters
- `Wire.py` - Source of wire parameters
- `InputParameter.py` - Source of input parameters
- `formula.py` - Circuit calculation functions (can use symbolic params)
- `globals.py` - Global instances (tech, cell, etc.)

## ⚡ Performance Tips

- **Symbolic mode**: Use for understanding, sensitivity, optimization setup
- **Numerical mode**: Use for high-speed simulation, large sweeps
- **Hybrid**: Create symbolic expression once, evaluate many times with different values

## 🤝 Contributing

To add new symbolic calculations:

1. Study existing examples in `example_symbolic_destiny.py`
2. Follow the pattern:
   - Create symbolic expression
   - Print symbolic form
   - Evaluate with concrete values
   - Show sensitivity if applicable
3. Add to examples or create new example file

## 📞 Questions?

- **"How do I use this?"** → Start with `QUICK_START_SYMBOLIC.md`
- **"What variables exist?"** → See `SYMBOLIC_MODELING_README.md`
- **"Is this like CACTI?"** → Yes! See `SYMBOLIC_WALKTHROUGH.md`
- **"Can I see examples?"** → Run `example_symbolic_destiny.py`
- **"How to integrate?"** → See `SYMBOLIC_MODELING_COMPLETE.md` (Integration section)

---

**Last Updated**: Implementation complete
**Status**: ✅ Ready to use
**Maintainer**: See base_parameters.py file header
