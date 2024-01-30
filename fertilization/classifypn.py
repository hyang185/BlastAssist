import numpy as np

from embryovision.localfolders import embryovision_folder


def measure_pronuclei_count(pronuclei_detections):
    """
    Parameters
    ----------
    pronuclei_dataset: AnnotatedDataset
        Pronuclei detection results for the embryo.

    Returns
    -------
    probabilities: dict
        Probability that the embryo has 0, 1, 2, or >=3 pronuclei.
        These probabilities are weighted such that they do not include
        the prior.

    See Also
    --------
    measure_pronuclei_fade_frame
    measure_pronuclei_appear_frame
    """
    if len(pronuclei_detections.iterate_over_images()) < 3:
        msg = 'Pronuclei detector must have run for at least 3 frames.'
        raise InsufficientDetectionsError(msg)
    confidences = strip_to_confidences(pronuclei_detections)
    feature = get_max_expected_pn(confidences)
    probabilities = predict_probability(feature)
    return probabilities


def measure_pronuclei_fade_frame(pronuclei_detections):
    """
    Parameters
    ----------
    pronuclei_dataset: AnnotatedDataset
        Pronuclei detection results for the embryo.

    Returns
    -------
    last_frame: {`embryovision.managedata.ImageInfo`, None}
        The last frame in which any pronuclei were detected.
        If not pronuclei were detected, returns None

    See Also
    --------
    measure_pronuclei_count
    measure_pronuclei_appear_frame
    """
    last_frame = None
    for image in pronuclei_detections.iterate_over_images():
        probs = [
            confidence_to_probability(d['confidence'])
            for d in image.annotation]
        if len(probs) > 0 and max(probs) > 0.5:
            last_frame = image.info
    return last_frame


def measure_pronuclei_appear_frame(pronuclei_detections):
    """
    Parameters
    ----------
    pronuclei_dataset: AnnotatedDataset
        Pronuclei detection results for the embryo.

    Returns
    -------
    last_frame: {`embryovision.managedata.ImageInfo`, None}
        The last frame in which any pronuclei were detected.
        If not pronuclei were detected, returns None

    See Also
    --------
    measure_pronuclei_count
    measure_pronuclei_fade_frame
    """
    first_frame = None
    for image in sorted(pronuclei_detections.iterate_over_images())[::-1]:
        probs = [
            confidence_to_probability(d['confidence'])
            for d in image.annotation]
        if len(probs) > 0 and max(probs) > 0.5:
            first_frame = image.info
    return first_frame


def strip_to_confidences(dataset):
    out = list()
    for i in dataset.iterate_over_images():
        these_confidences = [d['confidence'] for d in i.annotation]
        out.append(np.asarray(these_confidences))
    return out


def get_max_expected_pn(pn_confidences, percent=95):
    """
    Parameters
    ----------
    pn_confidences: ragged nested list
        The confidence scores, returned by the classifier, for the
        number of PN in the image. If image t has c pronuclei
        candidates, then pn_confidences[t] should be a length c
        list-like.

    Returns
    -------
    score : float
        The 95% of the smoothed expected PN count per image.
    """
    expected = list()
    for these_confidences in pn_confidences:
        probs = confidence_to_probability(these_confidences)
        expected.append(probs.sum())
    # smoothing and taking the 95% perecntile
    expected = np.asarray(expected)
    kernel = np.ones(3)
    kernel /= kernel.sum()
    smoothed = np.convolve(expected, kernel, mode='valid')
    return np.percentile(smoothed, percent)


def predict_probability(feature):
    x = np.squeeze(feature)
    if x.shape != ():
        raise ValueError('`feature` must be scalar')

    # hard-coded from fits, to avoid dealing with sklearn dependencies:
    slope = np.array([
        -3.0370291895216797,
        -1.0264109025618686,
        +1.2937132919064929,
        +2.769726800177056])
    intercept = np.array([
        +3.969540711190046,
        +2.1776435387113597,
        -1.4436112650130675,
        -4.703572984888345,
        ])

    z = slope * x + intercept
    probs = np.exp(z - z.max())
    probs /= probs.sum()

    keys = ['0', '1', '2', '3+']
    return {k: v for k, v in zip(keys, probs)}


def confidence_to_probability(confidence):
    confidence = np.clip(confidence, 0.03, 1.0)
    x = logit(confidence)
    # hard-coded from the fits:
    # MAP model order: quadratic rescaling of parameters
    # params = np.array([-2.96717424,  0.41597676,  0.01696053])
    # linear rescaling MAP fit:
    # FIXME should we do quadratic or linear rescaling?
    # (log-evidence for quadratic is -1911.13
    #               for linear    is -1938.26
    # which **strongly** prefers quadratic
    params = np.array([-2.74358973, 0.5819833])
    logit_prob = np.polynomial.chebyshev.chebval(x, params)
    return logistic(logit_prob)


def logistic(x):
    return 1.0 / (1 + np.exp(-x))


def logit(p):
    p = np.asarray(p)
    return np.log(p / (1 - p))


class InsufficientDetectionsError(ValueError):
    pass
