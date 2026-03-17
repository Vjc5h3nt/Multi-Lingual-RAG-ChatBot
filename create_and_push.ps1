# Create and push to GitHub repository
param(
    [string]$RepoName = "Multi-Lingual-RAG-ChatBot",
    [Parameter(Mandatory = $true)]
    [string]$Pat,
    [string]$Username = "Ganesh7171"
)

if (-not $Pat) {
    Write-Host "A GitHub Personal Access Token must be provided with -Pat." -ForegroundColor Red
    exit 1
}

# Step 1: Create repository on GitHub
Write-Host "Creating repository '$RepoName' on GitHub..." -ForegroundColor Cyan
$headers = @{
    Authorization = "token $Pat"
    "Content-Type" = "application/json"
}

$repoBody = @{
    name = $RepoName
    description = "Multi-language RAG chatbot with AWS Bedrock integration"
    private = $false
} | ConvertTo-Json

try {
    $repoResponse = Invoke-RestMethod -Method POST `
        -Headers $headers `
        -Uri "https://api.github.com/user/repos" `
        -Body $repoBody
    
    $repoUrl = $repoResponse.html_url
    $repoCloneUrl = $repoResponse.clone_url
    Write-Host "✓ Repository created successfully!" -ForegroundColor Green
    Write-Host "  URL: $repoUrl" -ForegroundColor Green
    
} catch {
    Write-Host "✗ Failed to create repository: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Step 2: Configure git remote and push
Write-Host "`nConfiguring git remote and pushing..." -ForegroundColor Cyan

# Get username for clone URL with token
$repoCloneWithToken = "https://${Pat}@github.com/${Username}/${RepoName}.git"

# Reset remote and push
git remote remove origin 2>$null
git remote add origin $repoCloneWithToken

try {
    git push -u origin main
    Write-Host "✓ Repository pushed successfully!" -ForegroundColor Green
    
    # Replace remote URL to remove token
    git remote set-url origin "https://github.com/${Username}/${RepoName}.git"
    Write-Host "✓ Remote URL configured (token removed)" -ForegroundColor Green
    
} catch {
    Write-Host "✗ Failed to push: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ All done! Repository is ready at: $repoUrl" -ForegroundColor Green
