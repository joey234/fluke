#!/usr/bin/env python
"""
Test script to verify all imports and basic functionality work.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from consolidated_analysis import FLUKEAnalyzer
        print("✓ FLUKEAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Error importing FLUKEAnalyzer: {e}")
        return False
    
    try:
        from statistical_analysis import StatisticalAnalyzer
        print("✓ StatisticalAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Error importing StatisticalAnalyzer: {e}")
        return False
    
    try:
        from generate_latex_tables import LaTeXTableGenerator
        print("✓ LaTeXTableGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Error importing LaTeXTableGenerator: {e}")
        return False
    
    try:
        from visualization import FLUKEVisualizer
        print("✓ FLUKEVisualizer imported successfully")
    except ImportError as e:
        print(f"❌ Error importing FLUKEVisualizer: {e}")
        return False
    
    try:
        from utils import Config, FileOrganizer
        print("✓ Utils imported successfully")
    except ImportError as e:
        print(f"❌ Error importing utils: {e}")
        return False
    
    return True


def test_basic_initialization():
    """Test basic initialization of key components."""
    print("\nTesting basic initialization...")
    
    try:
        from consolidated_analysis import FLUKEAnalyzer
        analyzer = FLUKEAnalyzer("../")
        print("✓ FLUKEAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing FLUKEAnalyzer: {e}")
        return False
    
    try:
        from statistical_analysis import StatisticalAnalyzer
        stat_analyzer = StatisticalAnalyzer("../")
        print("✓ StatisticalAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing StatisticalAnalyzer: {e}")
        return False
    
    try:
        from generate_latex_tables import LaTeXTableGenerator
        latex_gen = LaTeXTableGenerator("../")
        print("✓ LaTeXTableGenerator initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing LaTeXTableGenerator: {e}")
        return False
    
    try:
        from utils import FileOrganizer
        organizer = FileOrganizer("test_output")
        print("✓ FileOrganizer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing FileOrganizer: {e}")
        return False
    
    return True


def test_data_loading():
    """Test that we can load some data."""
    print("\nTesting data loading...")
    
    try:
        from consolidated_analysis import FLUKEAnalyzer
        analyzer = FLUKEAnalyzer("../")
        
        # Try to load one task
        tasks_to_try = ['coref', 'dialogue', 'ner', 'sa']
        
        for task in tasks_to_try:
            try:
                df = analyzer.load_results(task)
                if not df.empty:
                    print(f"✓ Successfully loaded {task} data: {len(df)} records")
                    return True
                else:
                    print(f"⚠️  No data found for {task}")
            except Exception as e:
                print(f"⚠️  Error loading {task}: {e}")
                continue
        
        print("❌ Could not load any task data")
        return False
        
    except Exception as e:
        print(f"❌ Error during data loading test: {e}")
        return False


def main():
    """Run all tests."""
    print("="*50)
    print("FLUKE Analysis Import Test")
    print("="*50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test initialization
    if not test_basic_initialization():
        success = False
    
    # Test data loading
    if not test_data_loading():
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 ALL TESTS PASSED! Ready to run full analysis.")
    else:
        print("❌ SOME TESTS FAILED. Please check the issues above.")
    print("="*50)
    
    return success


if __name__ == "__main__":
    main()