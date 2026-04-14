/**
 * Upper Valley MH Finder — Google Apps Script Web App
 * ====================================================
 * Receives POST data from the site and appends rows to a Google Sheet.
 * Every search and every star rating gets its own timestamped row.
 *
 * SETUP (one-time, ~5 minutes)
 * ----------------------------
 * 1. Go to https://sheets.google.com → create a new spreadsheet.
 *    Name it "UV MH Finder Data" (or anything you like).
 *
 * 2. Open Extensions → Apps Script.
 *    Delete the placeholder code and paste this entire file.
 *
 * 3. Click Deploy → New deployment:
 *      Type        : Web app
 *      Execute as  : Me
 *      Who has access: Anyone
 *    Click Deploy → copy the Web app URL (ends in /exec).
 *
 * 4. Open index.html in your editor and set:
 *      const SUBMIT_ENDPOINT = 'https://script.google.com/macros/s/YOUR_ID/exec';
 *
 * 5. Commit & push — data will flow into your sheet immediately.
 *
 * SHEET COLUMNS
 * -------------
 * A: Timestamp      B: Type (search/rating)
 * C: Provider ID    D: Star rating (ratings only)
 * E: Match score    F: Zip code
 * G: Insurance      H: Age group
 * I: Telehealth     J: Urgency
 * K: Session format L: Language
 * M: Gender pref    N: Concerns (comma-separated)
 * O: Modalities     P: Cultural prefs
 * Q: Accessibility  R: Raw JSON (full payload)
 */

const SHEET_NAME = 'Responses';

function doPost(e) {
  try {
    const payload = JSON.parse(e.postData.contents || '{}');
    appendRow(payload);
    return jsonResponse({ ok: true });
  } catch (err) {
    return jsonResponse({ ok: false, error: err.message });
  }
}

// Also handle GET so you can test the endpoint in a browser
function doGet(e) {
  return jsonResponse({ ok: true, message: 'UV MH Finder endpoint is live.' });
}

function appendRow(payload) {
  const ss    = SpreadsheetApp.getActiveSpreadsheet();
  let   sheet = ss.getSheetByName(SHEET_NAME);

  // Create the sheet + header row on first use
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
    sheet.appendRow([
      'Timestamp', 'Type', 'Provider ID', 'Rating', 'Score',
      'Zip', 'Insurance', 'Age Group', 'Telehealth', 'Urgency',
      'Session Format', 'Language', 'Gender Pref',
      'Concerns', 'Modalities', 'Cultural', 'Accessibility',
      'Raw JSON',
    ]);
    // Freeze header and auto-resize
    sheet.setFrozenRows(1);
    sheet.getRange(1, 1, 1, 18).setFontWeight('bold');
  }

  const f = payload.filters || {};
  const arr = v => Array.isArray(v) ? v.join(', ') : (v || '');

  sheet.appendRow([
    payload.ts          || new Date().toISOString(),
    payload.type        || '',
    payload.providerId  || '',
    payload.rating      != null ? payload.rating : '',
    payload.score       != null ? payload.score  : '',
    f.zip               || '',
    f.insurance         || '',
    f.ageGroup          || '',
    f.telehealth        || '',
    f.urgency           || '',
    f.sessionFormat     || '',
    f.language          || '',
    f.providerGender    || '',
    arr(f.concerns),
    arr(f.modalities),
    arr(f.cultural),
    arr(f.accessibility),
    JSON.stringify(payload),
  ]);
}

function jsonResponse(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
