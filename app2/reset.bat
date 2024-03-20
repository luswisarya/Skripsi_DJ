@echo off
set "baseDir=%cd%"
set "itemsToDelete=dataset classifier.pkl"

for %%i in (%itemsToDelete%) do (
    if exist "%baseDir%\%%i" (
        rd /s /q "%baseDir%\%%i" 2>nul || del /q "%baseDir%\%%i"
        echo "%%i deleted."
    ) else (
        echo "%%i not found."
    )
)
