#include "main_func.h"

extern "C" void app_main(void) {
  setup();
  while (true) {
    loop();
  }
}