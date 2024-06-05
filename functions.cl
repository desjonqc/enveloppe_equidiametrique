__kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g, int n)
{
  int gid = get_global_id(0);
  int gid2 = get_global_id(1);
  res_g[gid * n + gid2] = a_g[gid * n + gid2] + b_g[gid * n + gid2];
}

__kernel void methode_1(__global int *matrice, __global const float *delta, const int offset, const int taille, const int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int i2 = (int) delta[k * 3] + offset;
    int j2 = (int) delta[k * 3 + 1] + offset;
    float r = delta[k * 3 + 2];
    if (length((float2)(i - i2, j - j2)) <= r)
    {
        matrice[i * taille + j] = 1;
    }
}

__kernel void calc_delta(__global float *delta, const int k_constant)
{
    int k = get_global_id(0);
    int i = delta[k * 3];
    int j = delta[k * 3 + 1];
    int i2 = delta[k_constant * 3];
    int j2 = delta[k_constant * 3 + 1];
    delta[k * 3 + 2] = max(delta[k * 3 + 2], length((float2)(i - i2, j - j2)));
}

__kernel void calc_delta_eff(__global float *positions, __global float *delta_matrix, const int N) {
    int k0 = get_global_id(0);
    int k1 = get_global_id(1);
    int i = positions[k0 * 2];
    int j = positions[k0 * 2 + 1];

    int i2 = positions[k1 * 2];
    int j2 = positions[k1 * 2 + 1];

    delta_matrix[k0 * N + k1] = length((float2)(i - i2, j - j2));
}

__kernel void format_delta(__global float *delta, __global float *positions, __global float *delta_max, const int N) {
    int k = get_global_id(0);
    int i = positions[k * 2];
    int j = positions[k * 2 + 1];
    delta[k * 3] = i;
    delta[k * 3 + 1] = j;
    delta[k * 3 + 2] = delta_max[k];
}

__kernel void reduce_matrix(__global float *matrix, __global float *reduced_matrix, const int matrix_len, const int reduced_matrix_len, const int precision, const float step)
{
    int i = get_global_id(0) * step;
    int j = get_global_id(1) * step;
    float moyenne = 0;
    float count = 0;
    for (int k = max(0, (int) (i - precision)); k < min(matrix_len - 1, (int) (i + step + precision)); k++)
    {
        for (int l = max(0, (int) (j - precision)); l < min(matrix_len - 1, (int) (j + step + precision)); l++)
        {
            moyenne += matrix[k * matrix_len + l];
            count++;
        }
    }
    moyenne /= count;
    if (moyenne != 1 && moyenne > 0)
    {
        reduced_matrix[get_global_id(0) * reduced_matrix_len + get_global_id(1)] = 1;
    } else {
        reduced_matrix[get_global_id(0) * reduced_matrix_len + get_global_id(1)] = 0;
    }
}