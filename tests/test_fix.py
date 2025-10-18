# test_fix.py - Test the specific fix
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_artifact_fix():
    """Test if the ModelTrainerArtifact fix works"""
    try:
        from src.insurance_charges.entity.artifact_entity import ModelTrainerArtifact, RegressionMetricArtifact
        
        # Test creating artifact with trained_model
        metric_artifact = RegressionMetricArtifact(r2_score=0.85, rmse=1000, mae=800)
        model_artifact = ModelTrainerArtifact(
            trained_model_file_path="test.pkl",
            metric_artifact=metric_artifact,
            model_name="RandomForest",
            feature_count=10,
            trained_model="mock_model"  # This should now work
        )
        
        print(f"✅ ModelTrainerArtifact created successfully: {model_artifact}")
        print(f"✅ trained_model attribute: {model_artifact.trained_model}")
        return True
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        return False

if __name__ == "__main__":
    if test_artifact_fix():
        print("\n🎯 Fix applied successfully! Now run the comprehensive test again.")
    else:
        print("\n⚠️ Fix needs adjustment.")