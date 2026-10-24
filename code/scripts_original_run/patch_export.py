import pathlib
p = pathlib.Path("/root/autodl-tmp/PF-RPN/tools/inference/export_proposals.py")
s = p.read_text(encoding="utf-8")

old = """    if save_roi and bboxes.numel() > 0:
        roi_feats = model.extract_roi_features(
            inputs, bboxes, modality_mask=modality_mask)
    if roi_per_modality_flag and bboxes.numel() > 0:
        roi_per_modality = extract_per_modality_roi(
            model, inputs, bboxes, output_size=roi_size)
"""
new = """    # --- RoI frame fix (2026-10-09) -------------------------------------
    # `rescale=True` returns prediction boxes in the ORIGINAL image frame, but
    # the feature maps come from the RESIZED model input.  RoI-Align must use
    # model-input-frame boxes; otherwise every RoI is taken from the wrong
    # location (shifted toward the origin by 1/scale_factor).
    roi_boxes = bboxes
    if rescale and bboxes.numel() > 0:
        _sf = data_samples[0].scale_factor
        if torch.is_tensor(_sf):
            _sf = _sf.to(device=bboxes.device, dtype=bboxes.dtype)
        else:
            _sf = torch.as_tensor(_sf, dtype=bboxes.dtype, device=bboxes.device)
        roi_boxes = bboxes.clone()
        roi_boxes[:, [0, 2]] = roi_boxes[:, [0, 2]] / _sf[0]
        roi_boxes[:, [1, 3]] = roi_boxes[:, [1, 3]] / _sf[1]
    if save_roi and bboxes.numel() > 0:
        roi_feats = model.extract_roi_features(
            inputs, roi_boxes, modality_mask=modality_mask)
    if roi_per_modality_flag and bboxes.numel() > 0:
        roi_per_modality = extract_per_modality_roi(
            model, inputs, roi_boxes, output_size=roi_size)
"""
assert old in s, "roi anchor not found"
s = s.replace(old, new, 1)

anchor = "        roi_size=int(args.roi_size),"
assert anchor in s, "manifest anchor not found"
s = s.replace(anchor, anchor + "\n        bbox_frame='original',\n        roi_frame='model_input',\n        rescale=bool(args.rescale),", 1)

anchor2 = "'(stride 8) in the model input frame, extracted before '"
assert anchor2 in s, "notes anchor not found"
p.write_text(s, encoding="utf-8")
print("PATCHED")