type SaveMode = 'album' | 'files';

interface FileSystemWritableFileStreamLike {
  write(data: Blob): Promise<void>;
  close(): Promise<void>;
}

interface FileSystemFileHandleLike {
  createWritable(): Promise<FileSystemWritableFileStreamLike>;
}

interface FileSystemDirectoryHandleLike {
  getDirectoryHandle(name: string, options?: { create?: boolean }): Promise<FileSystemDirectoryHandleLike>;
  getFileHandle(name: string, options?: { create?: boolean }): Promise<FileSystemFileHandleLike>;
}

interface DirectoryPickerWindow extends Window {
  showDirectoryPicker?: (options?: { id?: string; mode?: 'read' | 'readwrite'; startIn?: string }) => Promise<FileSystemDirectoryHandleLike>;
}

const APP_FOLDER_NAME = 'ComfyUI Mobile';
const DEFAULT_MODEL_FOLDER = 'Z-Image Turbo';

function getSafeFilenameFromUrl(src: string, fallback: string): string {
  try {
    const url = new URL(src, window.location.href);
    const filename = url.searchParams.get('filename') || decodeURIComponent(url.pathname.split('/').pop() || '');
    return sanitizePathSegment(filename || fallback);
  } catch {
    return sanitizePathSegment(fallback);
  }
}

function sanitizePathSegment(value: string): string {
  return value
    .replace(/[\\/:*?"<>|]+/g, '-')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 120) || 'image.png';
}

function inferModelFolder(filename: string): string {
  const lower = filename.toLowerCase();
  if (lower.includes('z_image') || lower.includes('z-image') || lower.includes('zimage')) return DEFAULT_MODEL_FOLDER;
  const stem = filename.replace(/\.[^.]+$/, '');
  const prefix = stem
    .replace(/[_-]?\d{4,}.*$/, '')
    .replace(/[_-]?seed[_-]?\d+.*$/i, '')
    .replace(/[_-]+/g, ' ')
    .trim();
  return sanitizePathSegment(prefix || DEFAULT_MODEL_FOLDER);
}

async function fetchAsFile(src: string, filename: string): Promise<File> {
  const response = await fetch(src, { credentials: 'same-origin' });
  if (!response.ok) throw new Error(`Failed to fetch image (${response.status})`);
  const blob = await response.blob();
  const contentType = blob.type || response.headers.get('content-type') || 'image/png';
  return new File([blob], filename, { type: contentType });
}

function browserDownload(file: File, onDownloaded?: () => void) {
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = file.name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  onDownloaded?.();
}

async function shareFile(file: File, text: string): Promise<boolean> {
  const nav = navigator as Navigator & { canShare?: (data: ShareData) => boolean };
  if (!navigator.share) return false;
  const data: ShareData = { files: [file], title: file.name, text };
  if (nav.canShare && !nav.canShare(data)) return false;
  await navigator.share(data);
  return true;
}

async function saveFileToAppFolder(file: File, modelFolder: string): Promise<boolean> {
  const picker = (window as DirectoryPickerWindow).showDirectoryPicker;
  if (!picker) return false;
  const root = await picker({ id: 'comfyui-mobile-outputs', mode: 'readwrite', startIn: 'pictures' });
  const appDir = await root.getDirectoryHandle(APP_FOLDER_NAME, { create: true });
  const modelDir = await appDir.getDirectoryHandle(modelFolder, { create: true });
  const fileHandle = await modelDir.getFileHandle(file.name, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(file);
  await writable.close();
  return true;
}

export async function saveImage(
  src: string,
  filename: string = 'image.png',
  mode: SaveMode = 'album',
  onDownloaded?: (src: string) => void
) {
  const safeFilename = getSafeFilenameFromUrl(src, filename);
  try {
    const file = await fetchAsFile(src, safeFilename);
    if (mode === 'files') {
      const modelFolder = inferModelFolder(safeFilename);
      const savedToDirectory = await saveFileToAppFolder(file, modelFolder);
      if (!savedToDirectory) {
        const shared = await shareFile(file, `Save to Files → ${APP_FOLDER_NAME}/${modelFolder}`);
        if (!shared) browserDownload(file);
      }
    } else {
      const shared = await shareFile(file, 'Save image to Photos / album');
      if (!shared) browserDownload(file);
    }
    onDownloaded?.(src);
  } catch (err) {
    console.error(`Failed to save image (${mode}):`, err);
  }
}

export async function downloadImage(
  src: string,
  filename: string = 'image.png',
  onDownloaded?: (src: string) => void
) {
  await saveImage(src, filename, 'album', onDownloaded);
}

export async function saveImageToFiles(
  src: string,
  filename: string = 'image.png',
  onDownloaded?: (src: string) => void
) {
  await saveImage(src, filename, 'files', onDownloaded);
}

export async function downloadBatch(
  sources: string[],
  onDownloaded?: (src: string) => void,
  mode: SaveMode = 'album'
) {
  for (const src of sources) {
    await saveImage(src, 'image.png', mode, onDownloaded);
  }
}
