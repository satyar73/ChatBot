# Google Drive API Setup Guide

## 1. Create a Google Cloud Project
1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click "New Project"
4. Enter a name for your project and click "Create"
5. Once the project is created, select it from the project dropdown

## 2. Enable the Google Drive API
1. In your Google Cloud project, go to the "APIs & Services" > "Library" section
2. Search for "Google Drive API"
3. Click on "Google Drive API" in the search results
4. Click "Enable"

## 3. Create Service Account Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" and select "Service Account"
3. Enter a name and description for your service account, then click "Create"
4. For the role, you can select "Project" > "Viewer" for minimal permissions, then click "Continue"
5. (Optional) Add users who can use this service account if needed
6. Click "Done"

## 4. Generate and Download the Service Account Key
1. In the "Service Accounts" section, find the service account you just created
2. Click on the three dots menu at the end of the row and select "Manage Keys"
3. Click "Add Key" > "Create New Key"
4. Select "JSON" as the key type and click "Create"
5. The JSON key file will be downloaded to your computer - keep this secure!

## 5. Share Your Google Drive Files/Folders with the Service Account
1. Open the downloaded JSON key file and note the "client_email" value (it should look like `something@project-id.iam.gserviceaccount.com`)
2. Go to your Google Drive
3. Right-click on the folder you want to index and select "Share"
4. Add the service account email address and give it "Viewer" access
5. Click "Share"

## 6. Update Your Environment Variables

Add these variables to your .env file:

```
USE_GOOGLE_DRIVE=true
GOOGLE_DRIVE_CREDENTIALS_FILE=/path/to/your-credentials.json
GOOGLE_DRIVE_FOLDER_ID=optional-folder-id-to-start-from
```
The `GOOGLE_DRIVE_FOLDER_ID` is optional. If not provided, the indexer will start from the root of the Drive (all files shared with the service account).

## 7. Install Required Dependencies
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib python-docx PyPDF2
```

## 8. Run the Indexer
Now you can run your GoogleDriveIndexer:

## Troubleshooting
- **Permission issues**: Make sure the Google Drive files/folders are properly shared with the service account email
- **API not enabled**: Ensure the Google Drive API is enabled in your Google Cloud project
- **Missing dependencies**: Check that all required packages are installed
- **Invalid credentials**: Verify the path to your credentials file is correct and the file is valid

## Security Notes
- Keep your service account credentials secure - they provide access to your Google Drive files
- Use the principle of least privilege - only share the specific folders your application needs to access
- Consider using environment variables or a secrets manager rather than hardcoding paths in your code
