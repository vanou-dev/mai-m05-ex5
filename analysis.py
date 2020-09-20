#!/usr/bin/env python
# coding=utf-8


def CER(prediction, true_labels):
    """
    Calculates the classification error rate for an N-class classification problem


    Parameters
    ==========

    prediction : numpy.ndarray
        A 1D :py:class:`numpy.ndarray` containing your prediction

    true_labels : numpy.ndarray
          A 1D :py:class:`numpy.ndarray` containing the ground truth labels for
          the input array, organized in the same order.


    Returns
    =======

    CER : float
        The classification error rate

    """

    errors = (prediction != true_labels).sum()
    return errors / len(prediction)
