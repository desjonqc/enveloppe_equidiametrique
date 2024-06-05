import numpy as np

import pyopencl as cl


class CLField:
    """ Permet de manipuler des champs de données en OpenCL. """
    mf = cl.mem_flags

    def __init__(self, field: any):
        self.field = field

    def convert_to_cl(self, context: cl.Context, flags: int) -> cl.Buffer:
        return cl.Buffer(context, flags, hostbuf=self.field)

    def retrieve_from_cl(self, queue: cl.CommandQueue, buffer: cl.Buffer) -> np.ndarray:
        dest = np.empty_like(self.field)
        cl.enqueue_copy(queue, dest, buffer)
        return dest


class OpenCLProgram:
    """ Permet de charger un programme OpenCL et de l'exécuter. """
    def __init__(self, filename: str):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # Create a program from the kernel source code in functions.cl
        self.prg = cl.Program(self.ctx, open(filename).read()).build()

    def call_function(self, function_name: str, shape: tuple, *args):
        """
        Appelle une fonction OpenCL.
        :param function_name: Le nom de la fonction à appeler.
        :param shape: La forme du parallélisme.
        :param args: Les arguments de la fonction.
        """
        function = getattr(self.prg, function_name)
        function(self.queue, shape, None, *args)

    def close(self):
        """ Ferme le programme OpenCL. """
        self.queue.finish()

# program = OpenCLProgram("functions.cl")
# M1 = CLField(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32))
# M2 = CLField(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32))
#
# R = CLField(np.zeros((3, 3), dtype=np.float32))
#
# M1_buffer = M1.convert_to_cl(program.ctx, CLField.mf.READ_ONLY | CLField.mf.COPY_HOST_PTR)
# M2_buffer = M2.convert_to_cl(program.ctx, CLField.mf.READ_ONLY | CLField.mf.COPY_HOST_PTR)
# R_buffer = R.convert_to_cl(program.ctx, CLField.mf.WRITE_ONLY | CLField.mf.COPY_HOST_PTR)
#
# n_buffer = np.int32(3)
#
#
# program.call_function("sum", (3, 3), M1_buffer, M2_buffer, R_buffer, n_buffer)
#
# print(R.retrieve_from_cl(program.queue, R_buffer))
