# finds the STIM library (downloads it if it isn't present)
# set STIMLIB_PATH to the directory containing the stim subdirectory (the stim repository)

include(FindPackageHandleStandardArgs)

set(STIM_INCLUDE_DIR $ENV{STIMLIB_PATH})

find_package_handle_standard_args(STIM DEFAULT_MSG STIM_INCLUDE_DIR)

if(STIM_FOUND)
    set(STIM_INCLUDE_DIRS ${STIM_INCLUDE_DIR})
elseif(STIM_FOUND)
	#if the STIM library isn't found, download it
	file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/stimlib)	#remove the stimlib directory if it exists
	set(STIM_GIT "https://git.stim.ee.uh.edu/codebase/stimlib.git")
	execute_process(COMMAND git clone --depth 1 ${STIM_GIT} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	set(STIM_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/stimlib" CACHE TYPE PATH)
endif(STIM_FOUND)

find_package_handle_standard_args(STIM DEFAULT_MSG STIM_INCLUDE_DIR)
