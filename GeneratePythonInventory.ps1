<#
.SYNOPSIS
    Generates an inventory of Python files by concatenating their contents into a single text file.

.DESCRIPTION
    This script recursively searches for Python files in the current directory, excluding specified directories and optionally excluding test files. It writes the concatenated contents of these files into an output file with line numbers and headers for each file.

.PARAMETER ExcludeTests
    Optional switch to exclude test files (e.g., files starting with 'test_' or ending with '_test.py').

.EXAMPLE
    .\GeneratePythonInventory.ps1 -ExcludeTests
#>

param (
    [switch]$ExcludeTests
)

# Ensure the script runs on PowerShell 5+ for compatibility
$PSVersion = $PSVersionTable.PSVersion.Major
if ($PSVersion -lt 5) {
    Write-Error "This script requires PowerShell version 5 or higher."
    exit
}

# Set the output file name
$OutputFile = "python_files_inventory.txt"

# Initialize StreamWriter with UTF8 encoding
$stream = [System.IO.StreamWriter]::new($OutputFile, $false, [System.Text.Encoding]::UTF8)

try {
    # Write header
    $header = "// Python Files Concatenated on $(Get-Date -Format 'MM/dd/yyyy HH:mm:ss')`n// ----------------------------------------`n`n"
    $stream.WriteLine($header)

    # Define patterns to exclude test files if the ExcludeTests parameter is set
    if ($ExcludeTests) {
        $TestFilePatterns = @(
            '^test_.*\.py$',      # Files starting with 'test_'
            '.*_test\.py$',       # Files ending with '_test.py'
            '\\tests?\\',         # Files within 'test' or 'tests' directories
            '\\test_.*\\'         # Any other test directory patterns
        )
    }

    # Find all Python files recursively, excluding the venv directory and optionally excluding test files
    $PythonFiles = Get-ChildItem -Path . -Filter *.py -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -notmatch '\\venv\\' -and (
                -not $ExcludeTests -or (
                    $TestFilePatterns | ForEach-Object { $_ } | ForEach-Object { $_ -notmatch $_ }
                )
            )
        } |
        Sort-Object FullName

    # Alternatively, using more efficient filtering:
    $PythonFiles = Get-ChildItem -Path . -Filter *.py -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -notmatch '\\venv\\' -and (
                -not $ExcludeTests -or (
                    ($_.Name -notmatch '^test_.*\.py$') -and
                    ($_.Name -notmatch '.*_test\.py$') -and
                    ($_.FullName -notmatch '\\test(s)?\\')
                )
            )
        } |
        Sort-Object FullName

    # Process each file
    foreach ($File in $PythonFiles) {
        try {
            # Write file separator and information
            $stream.WriteLine("")
            $stream.WriteLine("// File: $($File.FullName)")
            $stream.WriteLine("// ----------------------------------------")

            # Initialize line counter
            $LineNumber = 1

            # Open StreamReader
            $reader = [System.IO.File]::OpenText($File.FullName)
            try {
                while (!$reader.EndOfStream) {
                    $line = $reader.ReadLine()
                    
                    # Safely concatenate the line number and the line content
                    $formattedLineNumber = "{0:D4}: " -f $LineNumber
                    $stream.WriteLine($formattedLineNumber + $line)
                    
                    $LineNumber++
                }
            }
            finally {
                $reader.Close()
            }
        }
        catch {
            # Log the error in Spanish as per original error messages
            $stream.WriteLine("// Error processing $($File.FullName): $_")
        }
    }

    # Write summary
    $stream.WriteLine("")
    $stream.WriteLine("// ----------------------------------------")
    $stream.WriteLine("// Total Python files found: $($PythonFiles.Count)")
}
finally {
    # Ensure the stream is closed
    $stream.Close()
}

# Print completion message
Write-Host "Inventory has been created in $OutputFile"

# Print found files if any
if ($PythonFiles.Count -gt 0) {
    Write-Host "Found files:"
    $PythonFiles | ForEach-Object { Write-Host $_.FullName }
} else {
    Write-Host "No Python files found."
}
