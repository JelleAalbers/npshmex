from npshmex import *
import npshmex


def square(x):
    return x ** 2


def square_2(bla):
    assert bla['y'] == 'something'
    return dict(x=bla['x']**2, y=bla['y'])


def nop(x):
    return x


def test_shm():
    shm_clear()

    with ProcessPoolExecutor() as ex:
        f = ex.submit(square, np.arange(10))
        assert f.result().sum() == 285

        # Temporary shm did not survive
        assert not len(SharedArray.list())

        x = shm_put(np.arange(10))
        f = ex.submit(square, x)
        assert f.result().sum() == 285

        # User-created shm does survive
        assert len(SharedArray.list()) == 1

        # But we can delete it
        assert shm_pop(x, keep=False).sum() == 45
        assert len(SharedArray.list()) == 0

        # Support for dicts of arrays
        inp = dict(x=np.arange(10), y='something')
        f = ex.submit(square_2, inp)
        result = f.result()
        assert isinstance(result, dict)
        assert result['x'].sum() == 285
        assert result['y'] == 'something'
        assert len(result) == 2

        assert len(SharedArray.list()) == 0

        # Support for structured arrays
        x = np.zeros(2, dtype=[('a', np.float), ('b', np.float)])
        x[0]['a'] = 42
        f = ex.submit(nop, x)
        result = f.result()
        assert result[0]['a'] == 42


class Wrapper:
    def __init__(self, bla, data):
        self.bla = bla
        self.data = data


def square_wrapper(w):
    w.data = w.data**2
    return w


def test_custom_dtype():

    register_array_wrapper(Wrapper, 'data')

    with ProcessPoolExecutor() as ex:
        f = ex.submit(square_wrapper, Wrapper(42, np.arange(10)))
        w = f.result()
        assert w.bla == 42
        np.testing.assert_array_equal(
            w.data,
            np.arange(10)**2)


if __name__ == '__main__':
    test_shm()
