# Sample script to install Python and pip under Windows
# Authors: Olivier Grisel, Jonathan Helmus and Kyle Kastner
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

$FSAVERAGE_URL = "https://staff.washington.edu/larsoner/fsaverage_min.zip"

Add-Type -AssemblyName System.IO.Compression.FileSystem
function Unzip
{
    param([string]$zipfile, [string]$outpath)
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

function DownloadExtractFsaverage () {
    $webclient = New-Object System.Net.WebClient
    $basedir = $pwd.Path + "\"
    $filepath = $basedir + "fsaverage_min.zip"
    # Download and retry up to 3 times in case of network transient errors.
    $url = $FSAVERAGE_URL
    Write-Host "Downloading" $url
    $retry_attempts = 2
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
    }
    if (Test-Path $filepath) {
        Write-Host "File saved at" $filepath
    } else {
        # Retry once to get the error message if any at the last try
        $webclient.DownloadFile($url, $filepath)
    }
    # Now we extract
    $subjects_dir = $basedir + "\subjects"
    Unzip $filepath $subjects_dir
}

function main () {
    DownloadExtractFsaverage
}

main