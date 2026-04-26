
REM Repository: projects/scml
echo Restoring: projects/scml
if exist "projects\scml\.git" (
    echo   Directory already exists, skipping...
) else (
    REM Create parent directory if needed
    if not exist "projects" mkdir "projects"

    REM Clone the repository
    git clone "git@github.com:yasserfarouk/scml.git" "projects\scml"
    if errorlevel 1 (
        echo   Failed to clone
    ) else (
        echo   Successfully cloned

        REM Checkout the original branch if not already on it
        cd "projects\scml"
        git checkout "master" 2>nul
        if errorlevel 1 (
            echo   Could not checkout branch: master
        ) else (
            echo   Checked out branch: master
        )
        cd ..\..
    )
)
echo.

REM Repository: projects/scml
echo Restoring: projects/scml
if exist "projects\scml\.git" (
    echo   Directory already exists, skipping...
) else (
    REM Create parent directory if needed
    if not exist "projects" mkdir "projects"
    
    REM Clone the repository
    git clone "git@github.com:yasserfarouk/scml.git" "projects\scml"
    if errorlevel 1 (
        echo   Failed to clone
    ) else (
        echo   Successfully cloned
        
        REM Checkout the original branch if not already on it
        cd "projects\scml"
        git checkout "master" 2>nul
        if errorlevel 1 (
            echo   Could not checkout branch: master
        ) else (
            echo   Checked out branch: master
        )
        cd ..\..
    )
)
echo.

