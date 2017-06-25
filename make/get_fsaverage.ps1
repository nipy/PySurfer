function main () {
    $webclient = New-Object System.Net.WebClient
    $basedir = $pwd.Path + "\"
    $filepath = $basedir + "fsaverage_min.zip"
    # Download and retry up to 3 times in case of network transient errors.
    $url = "http://faculty.washington.edu/larsoner/fsaverage_min.tar.gz"
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
    New-Item -ItemType directory -Path $subjects_dir
    $shell = new-object -com shell.application
    $zip = $shell.NameSpace($filepath)
    foreach($item in $zip.items())
    {
        $shell.Namespace($subjects_dir).copyhere($item)
    }
}

 main
