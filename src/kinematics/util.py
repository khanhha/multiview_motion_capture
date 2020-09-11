import numpy as np


def joints(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    joints : (J) ndarray
        Array of joint indices
    """
    return np.arange(len(parents), dtype=int)


def joints_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    joints : [ndarray]
        List of arrays of joint idices for
        each joint
    """
    return list(joints(parents)[:, np.newaxis])


def parents_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    parents : [ndarray]
        List of arrays of joint idices for
        the parents of each joint
    """
    return list(parents[:, np.newaxis])


def children_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    children : [ndarray]
        List of arrays of joint indices for
        the children of each joint
    """

    def joint_children(i):
        return [j for j, p in enumerate(parents) if p == i]

    return list(map(lambda j: np.array(joint_children(j)), joints(parents)))


def descendants_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    descendants : [ndarray]
        List of arrays of joint idices for
        the descendants of each joint
    """

    children = children_list(parents)

    def joint_descendants(i):
        return sum([joint_descendants(j) for j in children[i]], list(children[i]))

    return list(map(lambda j: np.array(joint_descendants(j)), joints(parents)))


def ancestors_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    ancestors : [ndarray]
        List of arrays of joint idices for
        the ancestors of each joint
    """

    decendants = descendants_list(parents)

    def joint_ancestors(i):
        return [j for j in joints(parents) if i in decendants[j]]

    return list(map(lambda j: np.array(joint_ancestors(j)), joints(parents)))


""" Mask Functions """


def mask(parents, filter):
    """
    Constructs a Mask for a give filter

    A mask is a (J, J) ndarray truth table for a given
    condition over J joints. For example there
    may be a mask specifying if a joint N is a
    child of another joint M.

    This could be constructed into a mask using
    `m = mask(parents, children_list)` and the condition
    of childhood tested using `m[N, M]`.

    Parameters
    ----------

    parents : (J) ndarray
        parents array

    filter : (J) ndarray -> [ndarray]
        function that outputs a list of arrays
        of joint indices for some condition

    Returns
    -------

    mask : (N, N) ndarray
        boolean truth table of given condition
    """
    m = np.zeros((len(parents), len(parents))).astype(bool)
    jnts = joints(parents)
    fltr = filter(parents)
    for i, f in enumerate(fltr): m[i, :] = np.any(jnts[:, np.newaxis] == f[np.newaxis, :], axis=1)
    return m


def joints_mask(parents): return np.eye(len(parents)).astype(bool)


def children_mask(parents): return mask(parents, children_list)


def parents_mask(parents): return mask(parents, parents_list)


def descendants_mask(parents): return mask(parents, descendants_list)


def ancestors_mask(parents): return mask(parents, ancestors_list)
