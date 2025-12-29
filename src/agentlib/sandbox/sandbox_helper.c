/*
 * sandbox_helper - Run commands with overlay filesystem using user namespaces
 *
 * Usage:
 *   sandbox_helper [--tar FILE] <overlay_target> -- <command> [args...]
 *
 * Creates temp overlay dirs, runs command in user+mount namespace, outputs tarball of changes.
 * If --tar FILE is given, writes tarball to FILE. Use - for stdout.
 *
 * Example:
 *   sandbox_helper --tar /tmp/changes.tar /home/jacob -- python agent.py
 *   sandbox_helper /home/jacob -- bash -c 'echo test > ~/file.txt'
 *
 * Compile: gcc -o sandbox_helper sandbox_helper.c
 * No root or SUID required - uses unprivileged user namespaces.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <ftw.h>

static const char *ALLOWED_BASES[] = {
    "/tmp/",
    "/home/",
    NULL
};

static char g_lower[PATH_MAX];
static char g_upper[PATH_MAX];
static char g_work[PATH_MAX];
static char g_tmpdir[PATH_MAX];

static int path_allowed(const char *path) {
    char resolved[PATH_MAX];
    if (realpath(path, resolved) == NULL) {
        return 0;
    }
    for (int i = 0; ALLOWED_BASES[i]; i++) {
        if (strncmp(resolved, ALLOWED_BASES[i], strlen(ALLOWED_BASES[i])) == 0) {
            return 1;
        }
    }
    return 0;
}

static int is_absolute(const char *path) {
    return path && path[0] == '/';
}

static int dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* Write to a file, used for uid_map/gid_map */
static int write_file(const char *path, const char *content) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -1;
    ssize_t len = strlen(content);
    ssize_t written = write(fd, content, len);
    close(fd);
    return (written == len) ? 0 : -1;
}

/* Recursive remove */
static int rm_callback(const char *path, const struct stat *st, int type, struct FTW *ftw) {
    (void)st; (void)type; (void)ftw;
    return remove(path);
}

static void cleanup_tmpdir(const char *path) {
    nftw(path, rm_callback, 64, FTW_DEPTH | FTW_PHYS);
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [--tar FILE] [--restore FILE] <target_dir> -- <command> [args...]\n", prog);
    fprintf(stderr, "  --tar FILE      Write tarball of changes to FILE (use - for stdout)\n");
    fprintf(stderr, "  --restore FILE  Pre-populate overlay with changes from FILE\n");
    fprintf(stderr, "  target_dir      Directory to overlay (e.g., /home/user)\n");
}

int main(int argc, char *argv[]) {
    uid_t real_uid = getuid();
    gid_t real_gid = getgid();
    const char *tar_file = NULL;
    const char *restore_file = NULL;
    int arg_idx = 1;

    /* Parse options */
    while (arg_idx < argc - 1) {
        if (strcmp(argv[arg_idx], "--tar") == 0 && arg_idx + 1 < argc) {
            tar_file = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--restore") == 0 && arg_idx + 1 < argc) {
            restore_file = argv[arg_idx + 1];
            arg_idx += 2;
        } else {
            break;
        }
    }

    if (argc < arg_idx + 3) {
        usage(argv[0]);
        return 1;
    }

    const char *target = argv[arg_idx];

    /* Find -- separator */
    int cmd_start = -1;
    for (int i = arg_idx + 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) {
            cmd_start = i + 1;
            break;
        }
    }
    if (cmd_start < 0 || cmd_start >= argc) {
        fprintf(stderr, "Error: missing -- separator or command\n");
        return 1;
    }

    /* Validate target */
    if (!is_absolute(target) || !path_allowed(target) || !dir_exists(target)) {
        fprintf(stderr, "Error: target must be absolute path in allowed directory\n");
        return 1;
    }

    /* Create temp directories */
    snprintf(g_tmpdir, sizeof(g_tmpdir), "/tmp/sandbox.%d.XXXXXX", getpid());
    if (mkdtemp(g_tmpdir) == NULL) {
        fprintf(stderr, "Error: mkdtemp failed: %s\n", strerror(errno));
        return 1;
    }

    snprintf(g_lower, sizeof(g_lower), "%s", target);
    snprintf(g_upper, sizeof(g_upper), "%s/upper", g_tmpdir);
    snprintf(g_work, sizeof(g_work), "%s/work", g_tmpdir);

    if (mkdir(g_upper, 0755) != 0 || mkdir(g_work, 0755) != 0) {
        fprintf(stderr, "Error: failed to create overlay dirs: %s\n", strerror(errno));
        cleanup_tmpdir(g_tmpdir);
        return 1;
    }

    /* Restore previous changes if requested */
    if (restore_file) {
        char tar_cmd[PATH_MAX * 2];
        snprintf(tar_cmd, sizeof(tar_cmd),
                 "tar -xf '%s' -C '%s' 2>/dev/null", restore_file, g_upper);
        int ret = system(tar_cmd);
        if (ret != 0) {
            fprintf(stderr, "Warning: failed to restore from %s (ret=%d)\n", restore_file, ret);
            /* Continue anyway - may be empty or missing */
        }
    }

    /* Fork - child runs sandboxed command, parent waits and generates tarball */
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "Error: fork failed: %s\n", strerror(errno));
        cleanup_tmpdir(g_tmpdir);
        return 1;
    }

    if (pid == 0) {
        /* Child: create user+mount namespace, mount overlay, exec command */

        /* Create user namespace and mount namespace */
        if (unshare(CLONE_NEWUSER | CLONE_NEWNS) != 0) {
            fprintf(stderr, "Error: unshare failed: %s\n", strerror(errno));
            _exit(1);
        }

        /* Write uid_map: map our uid to root (0) inside the namespace */
        char map[64];
        snprintf(map, sizeof(map), "0 %d 1\n", real_uid);
        if (write_file("/proc/self/uid_map", map) != 0) {
            fprintf(stderr, "Error: failed to write uid_map: %s\n", strerror(errno));
            _exit(1);
        }

        /* Disable setgroups (required before writing gid_map) */
        if (write_file("/proc/self/setgroups", "deny\n") != 0) {
            /* May fail on older kernels, continue anyway */
        }

        /* Write gid_map: map our gid to root (0) inside the namespace */
        snprintf(map, sizeof(map), "0 %d 1\n", real_gid);
        if (write_file("/proc/self/gid_map", map) != 0) {
            fprintf(stderr, "Error: failed to write gid_map: %s\n", strerror(errno));
            _exit(1);
        }

        /* Make mounts private */
        if (mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0) {
            fprintf(stderr, "Error: failed to make root private: %s\n", strerror(errno));
            _exit(1);
        }

        /* Mount the overlay */
        char options[PATH_MAX * 4];
        snprintf(options, sizeof(options),
                 "lowerdir=%s,upperdir=%s,workdir=%s",
                 g_lower, g_upper, g_work);

        if (mount("overlay", target, "overlay", 0, options) != 0) {
            fprintf(stderr, "Error: overlay mount failed: %s\n", strerror(errno));
            _exit(1);
        }

        /* Don't create .pyc files */
        setenv("PYTHONDONTWRITEBYTECODE", "1", 1);

        execvp(argv[cmd_start], &argv[cmd_start]);
        fprintf(stderr, "Error: exec failed: %s\n", strerror(errno));
        _exit(1);
    }

    /* Parent: wait for child, generate tarball, cleanup */
    int status;
    waitpid(pid, &status, 0);

    /* Generate tarball of upper layer (the changes) */
    if (tar_file && strcmp(tar_file, "-") != 0) {
        char tar_cmd[PATH_MAX * 2];
        snprintf(tar_cmd, sizeof(tar_cmd),
                 "tar -cf '%s' -C '%s' . 2>/dev/null", tar_file, g_upper);
        system(tar_cmd);
    } else if (tar_file && strcmp(tar_file, "-") == 0) {
        /* Output tar to stdout */
        char tar_cmd[PATH_MAX * 2];
        snprintf(tar_cmd, sizeof(tar_cmd),
                 "tar -cf - -C '%s' .", g_upper);
        system(tar_cmd);
    }

    /* Cleanup */
    cleanup_tmpdir(g_tmpdir);

    /* Return child's exit status */
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 1;
}
