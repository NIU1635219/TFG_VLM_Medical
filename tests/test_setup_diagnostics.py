import pytest
from unittest.mock import patch, MagicMock
from src.utils.setup_diagnostics import perform_diagnostics

def test_perform_diagnostics():
    # Mock context
    class DiagnosticIssue:
        def __init__(self, name, fix_fn, fix_label):
            self.name = name
            self.fix_fn = fix_fn
            self.fix_label = fix_label

    ctx = {
        "DiagnosticIssue": DiagnosticIssue,
        "fix_folders": MagicMock(),
        "fix_uv": MagicMock(),
        "fix_libs": MagicMock(),
        "check_uv": MagicMock(return_value=True),
        "check_lms": MagicMock(return_value=True),
        "log": MagicMock(),
        "REQUIRED_LIBS": ["pytest"],
        "LIB_IMPORT_MAP": {"pytest": "pytest"}
    }
    
    with patch('src.utils.setup_diagnostics.os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('src.utils.setup_diagnostics.shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/uv"
            
            with patch('builtins.__import__') as mock_import:
                mock_import.return_value = MagicMock()
                
                report, issues = perform_diagnostics(ctx)
                
                assert len(report) > 0
                # If everything is OK, there should be no issues
                # Wait, there might be disk space issues if we don't mock shutil.disk_usage
                # Let's mock disk_usage
                with patch('src.utils.setup_diagnostics.shutil.disk_usage') as mock_disk:
                    mock_disk.return_value = MagicMock(free=100 * 1024 * 1024 * 1024) # 100GB
                    report, issues = perform_diagnostics(ctx)
                    assert len(issues) == 0

def test_perform_diagnostics_with_issues():
    class DiagnosticIssue:
        def __init__(self, name, fix_fn, fix_label):
            self.name = name
            self.fix_fn = fix_fn
            self.fix_label = fix_label

    ctx = {
        "DiagnosticIssue": DiagnosticIssue,
        "fix_folders": MagicMock(),
        "fix_uv": MagicMock(),
        "fix_libs": MagicMock(),
        "check_uv": MagicMock(return_value=False),
        "check_lms": MagicMock(return_value=False),
        "log": MagicMock(),
        "REQUIRED_LIBS": ["pytest"],
        "LIB_IMPORT_MAP": {"pytest": "pytest"}
    }
    
    with patch('src.utils.setup_diagnostics.os.path.exists') as mock_exists:
        mock_exists.return_value = False # Missing folders
        
        with patch('src.utils.setup_diagnostics.shutil.which') as mock_which:
            mock_which.return_value = None # Missing uv
            
            with patch('builtins.__import__') as mock_import:
                mock_import.side_effect = ImportError # Missing libs
                
                with patch('src.utils.setup_diagnostics.shutil.disk_usage') as mock_disk:
                    mock_disk.return_value = MagicMock(free=1 * 1024 * 1024 * 1024) # 1GB (low space)
                    
                    report, issues = perform_diagnostics(ctx)
                    
                    assert len(issues) > 0
                    issue_names = [i.name for i in issues]
                    assert "Missing Folders" in issue_names
                    assert "Missing uv" in issue_names
                    assert "Missing pytest" in issue_names
