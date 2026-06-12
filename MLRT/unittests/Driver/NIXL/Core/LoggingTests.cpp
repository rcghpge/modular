//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

struct ChildResult {
  int exitCode = -1;
  std::string stdoutText;
  std::string stderrText;
};

std::string readAllFromFd(int fd) {
  std::string output;
  char buffer[4096];
  while (true) {
    ssize_t bytesRead = read(fd, buffer, sizeof(buffer));
    if (bytesRead <= 0) {
      return output;
    }
    output.append(buffer, bytesRead);
  }
}

ChildResult runHelper(std::string_view mode,
                      std::optional<std::string_view> logLevel = std::nullopt) {
  ChildResult result;
  const char *helperPath = std::getenv("TEST_NIXL_LOG_HELPER_PATH");
  if (helperPath == nullptr) {
    ADD_FAILURE() << "TEST_NIXL_LOG_HELPER_PATH is not set";
    return result;
  }

  int stdoutPipe[2];
  if (pipe(stdoutPipe) != 0) {
    ADD_FAILURE() << "failed to create stdout pipe";
    return result;
  }
  int stderrPipe[2];
  if (pipe(stderrPipe) != 0) {
    ADD_FAILURE() << "failed to create stderr pipe";
    close(stdoutPipe[0]);
    close(stdoutPipe[1]);
    return result;
  }

  pid_t childPid = fork();
  if (childPid == -1) {
    ADD_FAILURE() << "failed to fork helper process";
    close(stdoutPipe[0]);
    close(stdoutPipe[1]);
    close(stderrPipe[0]);
    close(stderrPipe[1]);
    return result;
  }

  if (childPid == 0) {
    dup2(stdoutPipe[1], STDOUT_FILENO);
    dup2(stderrPipe[1], STDERR_FILENO);
    close(stdoutPipe[0]);
    close(stdoutPipe[1]);
    close(stderrPipe[0]);
    close(stderrPipe[1]);

    if (logLevel) {
      setenv("MODULAR_NIXL_LOG_LEVEL", logLevel->data(), 1);
    } else {
      unsetenv("MODULAR_NIXL_LOG_LEVEL");
    }

    execl(helperPath, helperPath, mode.data(), nullptr);
    _exit(127);
  }

  close(stdoutPipe[1]);
  close(stderrPipe[1]);

  result.stdoutText = readAllFromFd(stdoutPipe[0]);
  result.stderrText = readAllFromFd(stderrPipe[0]);

  close(stdoutPipe[0]);
  close(stderrPipe[0]);

  int status = 0;
  if (waitpid(childPid, &status, 0) != childPid) {
    ADD_FAILURE() << "waitpid failed for helper process";
    return result;
  }
  if (!WIFEXITED(status)) {
    ADD_FAILURE() << "helper process did not exit normally";
    return result;
  }
  result.exitCode = WEXITSTATUS(status);
  return result;
}

} // namespace

TEST(NixlLoggingTests, DefaultWarnSuppressesInfoLogs) {
  ChildResult result = runHelper("default-warn");

  EXPECT_EQ(result.exitCode, 0);
  EXPECT_EQ(result.stdoutText, "");
  EXPECT_NE(result.stderrText.find("warn-visible"), std::string::npos);
  EXPECT_EQ(result.stderrText.find("info-hidden"), std::string::npos);
}

TEST(NixlLoggingTests, ErrorLevelSuppressesWarnings) {
  ChildResult result = runHelper("error-only", "ERROR");

  EXPECT_EQ(result.exitCode, 0);
  EXPECT_EQ(result.stdoutText, "");
  EXPECT_NE(result.stderrText.find("error-visible"), std::string::npos);
  EXPECT_EQ(result.stderrText.find("warn-hidden"), std::string::npos);
}

TEST(NixlLoggingTests, DebugLevelEnablesDebugButNotTrace) {
  ChildResult result = runHelper("debug-enabled", "DEBUG");

  EXPECT_EQ(result.exitCode, 0);
  EXPECT_EQ(result.stdoutText, "");
  EXPECT_NE(result.stderrText.find("debug-visible"), std::string::npos);
  EXPECT_EQ(result.stderrText.find("trace-hidden"), std::string::npos);
}

TEST(NixlLoggingTests, DisabledDebugLogsDoNotEvaluateOperands) {
  ChildResult result = runHelper("short-circuit");

  EXPECT_EQ(result.exitCode, 0);
  EXPECT_EQ(result.stdoutText, "counter=0\n");
  EXPECT_EQ(result.stderrText.find("debug-side-effect"), std::string::npos);
}

TEST(NixlLoggingTests, FatalLogsRemainVisible) {
  ChildResult result = runHelper("fatal-visible", "FATAL");

  EXPECT_EQ(result.exitCode, 0);
  EXPECT_EQ(result.stdoutText, "");
  EXPECT_NE(result.stderrText.find("fatal-visible"), std::string::npos);
}
