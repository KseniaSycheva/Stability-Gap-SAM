import torch

from src.utils import checkattr, get_data_loader


def test_acc(
    model,
    dataset,
    batch_size=128,
    test_size=1024,
    verbose=True,
    context_id=None,
    allowed_classes=None,
    no_context_mask=False,
    **kwargs,
):
    """Evaluate accuracy (= proportion of samples classified correctly) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)"""

    # Get device-type / using cuda?
    device = model.device if hasattr(model, "device") else model._device()
    cuda = model.cuda if hasattr(model, "cuda") else model._is_on_cuda()

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Apply context-specific "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_context_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(context=context_id + 1)

    # Should output-labels be adjusted for allowed classes? (ASSUMPTION: [allowed_classes] has consecutive numbers)
    label_correction = (
        0
        if checkattr(model, "stream_classifier") or (allowed_classes is None)
        else allowed_classes[0]
    )

    # If there is a separate network per context, select the correct subnetwork
    if model.label == "SeparateClassifiers":
        model = getattr(model, "context{}".format(context_id + 1))
        allowed_classes = None

    # Loop over batches in [dataset]
    data_loader = get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for x, y in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -if the model is a "stream-classifier", add context
        if checkattr(model, "stream_classifier"):
            context_tensor = torch.tensor([context_id] * x.shape[0]).to(device)
        # -evaluate model (if requested, only on [allowed_classes])
        with torch.no_grad():
            if checkattr(model, "stream_classifier"):
                scores = model.classify(x.to(device), context=context_tensor)
            else:
                scores = model.classify(x.to(device), allowed_classes=allowed_classes)
        _, predicted = torch.max(scores.cpu(), 1)
        if model.prototypes and max(predicted).item() >= model.classes:
            # -in case of Domain-IL (or Task-IL + singlehead), collapse all corresponding domains to same class
            predicted = predicted % model.classes
        # -update statistics
        y = y - label_correction
        total_correct += (predicted == y).sum().item()
        total_tested += len(x)
    accuracy = total_correct / total_tested

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    if verbose:
        print("=> accuracy: {:.3f}".format(accuracy))
    return accuracy
