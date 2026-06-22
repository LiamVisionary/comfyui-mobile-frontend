import { describe, expect, it } from "vitest";
import {
  resolveUploadFolder,
  isOutputFileSelectable,
  sortOutputPickerFiles,
} from "@/components/InputControls/outputPickerUtils";

describe("ComboControl output picker helpers", () => {
  it("keeps image uploads in the configured image folder", () => {
    expect(resolveUploadFolder(false, "mask_inputs")).toBe("mask_inputs");
  });

  it("forces video uploads into the input folder", () => {
    expect(resolveUploadFolder(true, "mask_inputs")).toBe("input");
  });

  it("only allows images for image upload combos", () => {
    expect(isOutputFileSelectable("image", false)).toBe(true);
    expect(isOutputFileSelectable("video", false)).toBe(false);
    expect(isOutputFileSelectable("folder", false)).toBe(false);
  });

  it("only allows videos for video upload combos", () => {
    expect(isOutputFileSelectable("video", true)).toBe(true);
    expect(isOutputFileSelectable("image", true)).toBe(false);
    expect(isOutputFileSelectable("folder", true)).toBe(false);
  });

  it("sorts selectable image outputs by newest first", () => {
    const sorted = sortOutputPickerFiles([
      { id: "output/folder", name: "folder", type: "folder", date: 9999 },
      { id: "output/old.png", name: "old.png", type: "image", date: 1000 },
      { id: "output/new.png", name: "new.png", type: "image", date: 3000 },
      { id: "output/movie.mp4", name: "movie.mp4", type: "video", date: 5000 },
    ], false);

    expect(sorted.map((file) => file.name)).toEqual([
      "new.png",
      "old.png",
      "folder",
      "movie.mp4",
    ]);
  });

  it("sorts selectable video outputs by newest first for video inputs", () => {
    const sorted = sortOutputPickerFiles([
      { id: "output/image.png", name: "image.png", type: "image", date: 5000 },
      { id: "output/old.mp4", name: "old.mp4", type: "video", date: 1000 },
      { id: "output/new.mp4", name: "new.mp4", type: "video", date: 3000 },
    ], true);

    expect(sorted.map((file) => file.name)).toEqual([
      "new.mp4",
      "old.mp4",
      "image.png",
    ]);
  });
});
