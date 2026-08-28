param(
    [string]$PythonExe = "python",
    [int]$TimeoutSeconds = 2400,
    [string]$Llm = "gpt-3.5-turbo",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$OutRoot = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $RepoRoot "outputs\fig4_full_benchmark"
} else {
    [System.IO.Path]::GetFullPath($OutputDir)
}
$CompareRoot = Join-Path $RepoRoot "autologic"
$SageRoot = Join-Path $RepoRoot "legacy_sage"
$Overlay = Join-Path $RepoRoot "compat"
$Logs = Join-Path $OutRoot "logs"
$Tables = Join-Path $OutRoot "tables"
$Raw = Join-Path $OutRoot "raw"

New-Item -ItemType Directory -Force -Path $Logs, $Tables, $Raw | Out-Null
Set-Content -Path (Join-Path $OutRoot "run_pid.txt") -Value $PID -Encoding UTF8

if (-not $env:OPENAI_BASE_URL -or -not $env:OPENAI_API_KEY) {
    throw "OPENAI_BASE_URL and OPENAI_API_KEY must be set in the runner environment."
}

$StatusPath = Join-Path $Tables "run_status.csv"
$JobsPath = Join-Path $Tables "jobs_manifest.csv"
$ManifestPath = Join-Path $Tables "run_manifest.json"

$settings = @{
    classification_binary_sage = @{ name = "f15_m15_p8"; f = 15; m = 15; p = 8; route = "sageloop_binary_cached_prediction" }
    classification_binary_compare = @{ name = "f15_m15_p8"; f = 15; m = 15; p = 8; route = "compare_binary_cached_prediction" }
    classification_multiclass_compare = @{ name = "f15_m15_p8"; f = 15; m = 15; p = 8; route = "compare_multiclass" }
    regression_sage = @{ name = "f10_m7"; f = 10; m = 7; p = ""; route = "sageloop_regression_cached_prediction" }
    regression_compare = @{ name = "f10_m7_p5"; f = 10; m = 7; p = 5; route = "compare_regression_cached_prediction_csv_fallback" }
    clustering_sage = @{ name = "f15_m15_p8"; f = 15; m = 15; p = 8; route = "sageloop_clustering_cached_labels" }
}

$manifest = [ordered]@{
    created_at = (Get-Date).ToString("s")
    python = $PythonExe
    llm = $Llm
    timeout_seconds_per_job = $TimeoutSeconds
    seeds = @(42,43,44,45,46)
    api_base = $env:OPENAI_BASE_URL
    api_key_written_to_files = $false
    settings = $settings
    notes = @(
        "Each dataset seed is launched as an independent process.",
        "Jobs exceeding TimeoutSeconds are killed, marked timeout, and the runner continues.",
        "credit-g workbook row is run through ds_credit for the SAGE binary route.",
        "forest workbook row is run through forest-fires for CSV regression fallback.",
        "puma8nh workbook row is run through puma8NH for CSV regression fallback.",
        "No raw API key is written to this manifest or logs by this runner."
    )
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding UTF8

function Quote-Arg([string]$value) {
    '"' + ($value -replace '"', '\"') + '"'
}

$jobs = New-Object System.Collections.Generic.List[object]

function Add-JobSpec(
    [string]$Framework,
    [string]$Task,
    [string]$Dataset,
    [string]$DatasetArg,
    [int]$Seed,
    [hashtable]$Setting,
    [string]$Workdir,
    [string]$Script,
    [string[]]$Argv,
    [string]$Notes
) {
    $jobs.Add([ordered]@{
        framework = $Framework
        task = $Task
        dataset = $Dataset
        dataset_arg = $DatasetArg
        seed = $Seed
        setting = $Setting.name
        route = $Setting.route
        workdir = $Workdir
        script = $Script
        argv = $Argv
        notes = $Notes
    }) | Out-Null
}

$seeds = @(42,43,44,45,46)

$sageClassScript = Join-Path $SageRoot "ensemble\classification_ensemble\classification_auto_ensemble.py"
$compareClassScript = Join-Path $CompareRoot "ensemble\classification_ensemble\classification_auto_ensemble.py"
$compareMultiScript = Join-Path $CompareRoot "ensemble\multiclassification_ensemble\multiclassification_auto_ensemble.py"
$sageRegScript = Join-Path $SageRoot "ensemble\regression_ensemble\regression_auto_ensemble.py"
$compareRegScript = Join-Path $CompareRoot "ensemble\regression_ensemble\regression_auto_ensemble.py"
$sageClusterScript = Join-Path $SageRoot "ensemble\cluster_ensemble\cluster_auto_ensemble_cached_labels.py"

$sageBinary = @(
    @{dataset="cd1"; arg="cd1"},
    @{dataset="cc1"; arg="cc1"},
    @{dataset="ld1"; arg="ld1"},
    @{dataset="credit-g"; arg="ds_credit"},
    @{dataset="cc2"; arg="cc2"},
    @{dataset="cd2"; arg="cd2"},
    @{dataset="cf1"; arg="cf1"}
)
$compareBinary = @("adult","bank","blood","heart","pc1","pc3","tic-tac-toe")
$compareMulti = @("balance-scale","cmc","eucalyptus","jungle_chess","car","vehicle")

foreach ($seed in $seeds) {
    foreach ($d in $sageBinary) {
        $s = $settings.classification_binary_sage
        Add-JobSpec "SAGE-Loop" "classification" $d.dataset $d.arg $seed $s `
            (Join-Path $SageRoot "ensemble\classification_ensemble") $sageClassScript `
            @("-d",$d.arg,"-s","$seed","-e","1","-l",$Llm,"-f","$($s.f)","-m","$($s.m)","-p","$($s.p)","--enable_optimization","--enable_feedback") `
            "old-pkl binary classification via SAGE route"
    }
    foreach ($dataset in $compareBinary) {
        $s = $settings.classification_binary_compare
        Add-JobSpec "autoLOGIC_compare" "classification" $dataset $dataset $seed $s `
            (Join-Path $CompareRoot "ensemble\classification_ensemble") $compareClassScript `
            @("-d",$dataset,"-s","$seed","-e","1","-l",$Llm,"-f","$($s.f)","-m","$($s.m)","-p","$($s.p)","--enable_optimization","--enable_feedback","--stage_timeout_s","300") `
            "csv binary classification via compare route"
    }
    foreach ($dataset in $compareMulti) {
        $s = $settings.classification_multiclass_compare
        Add-JobSpec "autoLOGIC_compare" "classification" $dataset $dataset $seed $s `
            (Join-Path $CompareRoot "ensemble\multiclassification_ensemble") $compareMultiScript `
            @("-d",$dataset,"-s","$seed","-e","1","-l",$Llm,"-f","$($s.f)","-m","$($s.m)","-p","$($s.p)","--enable_optimization","--enable_feedback","--stage_timeout_s","300") `
            "multiclass classification via compare route"
    }
}

$sageReg = @(
    @{dataset="boston"; arg="boston"},
    @{dataset="concrete"; arg="concrete"},
    @{dataset="insurance"; arg="insurance"},
    @{dataset="crab"; arg="crab_age"},
    @{dataset="winequality"; arg="winequality"},
    @{dataset="california"; arg="california"}
)
$compareReg = @(
    @{dataset="bike"; arg="bike"},
    @{dataset="forest"; arg="forest-fires"},
    @{dataset="wind"; arg="wind"},
    @{dataset="puma8nh"; arg="puma8NH"}
)

foreach ($seed in $seeds) {
    foreach ($d in $sageReg) {
        $s = $settings.regression_sage
        Add-JobSpec "SAGE-Loop" "regression" $d.dataset $d.arg $seed $s `
            (Join-Path $SageRoot "ensemble\regression_ensemble") $sageRegScript `
            @("-d",$d.arg,"-s","$seed","-e","1","-l",$Llm,"-f","$($s.f)","-m","$($s.m)","--enable_feedback") `
            "pkl regression via SAGE route; SAGE path has no param_iterations argument"
    }
    foreach ($d in $compareReg) {
        $s = $settings.regression_compare
        Add-JobSpec "autoLOGIC_compare" "regression" $d.dataset $d.arg $seed $s `
            (Join-Path $CompareRoot "ensemble\regression_ensemble") $compareRegScript `
            @("-d",$d.arg,"-s","$seed","-e","1","-l",$Llm,"-f","$($s.f)","-m","$($s.m)","-p","$($s.p)","--enable_optimization","--enable_feedback","--stage_timeout_s","300") `
            "csv regression via compare route with fallback loader"
    }
}

$clusterDatasets = @("breast","glass","iris","students","seeds","cd2","ld1","cd1","ld2","cc3")
foreach ($seed in $seeds) {
    foreach ($dataset in $clusterDatasets) {
        $s = $settings.clustering_sage
        Add-JobSpec "SAGE-Loop" "clustering" $dataset $dataset $seed $s `
            (Join-Path $SageRoot "ensemble\cluster_ensemble") $sageClusterScript `
            @("-d",$dataset,"-s","$seed","-e","1","-l",$Llm,"-f","$($s.f)","-m","$($s.m)","-p","$($s.p)","--enable_optimization","--enable_feedback","--stage_timeout_s","300","--min_ensemble_ari","0.30","--top_k_values","1","2","3","5") `
            "cached-label clustering; top_k sweep recorded in stdout/jsonl"
    }
}

"job_id,framework,task,dataset,dataset_arg,seed,setting,route,status,exit_code,runtime_sec,stdout,stderr,notes" | Set-Content -Path $StatusPath -Encoding UTF8
"job_id,framework,task,dataset,dataset_arg,seed,setting,route,script,workdir,argv,notes" | Set-Content -Path $JobsPath -Encoding UTF8

for ($i = 0; $i -lt $jobs.Count; $i++) {
    $job = $jobs[$i]
    $jobId = "{0:D4}_{1}_{2}_{3}_seed{4}" -f ($i + 1), $job.task, $job.dataset, $job.framework.Replace("autoLOGIC_","").Replace("-",""), $job.seed
    $argString = ($job.argv -join " ")
    $manifestLine = '"{0}","{1}","{2}","{3}","{4}",{5},"{6}","{7}","{8}","{9}","{10}","{11}"' -f `
        $jobId,$job.framework,$job.task,$job.dataset,$job.dataset_arg,$job.seed,$job.setting,$job.route,$job.script,$job.workdir,($argString -replace '"','""'),($job.notes -replace '"','""')
    Add-Content -Path $JobsPath -Value $manifestLine -Encoding UTF8
}

for ($i = 0; $i -lt $jobs.Count; $i++) {
    $job = $jobs[$i]
    $jobId = "{0:D4}_{1}_{2}_{3}_seed{4}" -f ($i + 1), $job.task, $job.dataset, $job.framework.Replace("autoLOGIC_","").Replace("-",""), $job.seed
    $stdout = Join-Path $Logs "$jobId.stdout.log"
    $stderr = Join-Path $Logs "$jobId.stderr.log"

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $PythonExe
    $psi.WorkingDirectory = $job.workdir
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.Environment["PYTHONIOENCODING"] = "utf-8"
    $psi.Environment["PYTHONPATH"] = "$Overlay;$($psi.Environment["PYTHONPATH"])"
    $psi.Environment["FIG5_OPENAI_TIMEOUT_S"] = "120"
    $psi.Environment["OPENAI_BASE_URL"] = $env:OPENAI_BASE_URL
    $psi.Environment["OPENAI_API_KEY"] = $env:OPENAI_API_KEY

    $allArgs = @($job.script) + $job.argv
    $psi.Arguments = ($allArgs | ForEach-Object { Quote-Arg $_ }) -join " "

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = [System.Diagnostics.Process]::Start($psi)
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()
    $finished = $proc.WaitForExit($TimeoutSeconds * 1000)
    $status = "completed"
    if (-not $finished) {
        $status = "timeout"
        try { $proc.Kill($true) } catch { try { $proc.Kill() } catch {} }
        $proc.WaitForExit()
    }
    $sw.Stop()
    $outTask.Wait()
    $errTask.Wait()

    [System.IO.File]::WriteAllText($stdout, $outTask.Result, [System.Text.Encoding]::UTF8)
    [System.IO.File]::WriteAllText($stderr, $errTask.Result, [System.Text.Encoding]::UTF8)
    $exitCode = if ($finished) { $proc.ExitCode } else { -999 }
    if ($finished -and $exitCode -ne 0) { $status = "failed" }

    $line = '"{0}","{1}","{2}","{3}","{4}",{5},"{6}","{7}","{8}",{9},{10},"{11}","{12}","{13}"' -f `
        $jobId,$job.framework,$job.task,$job.dataset,$job.dataset_arg,$job.seed,$job.setting,$job.route,$status,$exitCode,[math]::Round($sw.Elapsed.TotalSeconds,2),$stdout,$stderr,($job.notes -replace '"','""')
    Add-Content -Path $StatusPath -Value $line -Encoding UTF8
}
