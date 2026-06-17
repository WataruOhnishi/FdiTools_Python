%EXPORT_ALL_FIGS - run every example and export its figures to Examples/plot.
% Paste this whole file into the MATLAB command window, or just run:
%   export_all_figs
%
% Each example begins with "clear all", so this script uses NO variables that
% must survive between examples (every savefigs argument is a literal string).
% "pause off" lets examples that contain pause (e.g. Step_6) run unattended.
%
% Requirements: the FdiTools src on the path (addpath(genpath('../src'))) and
% the current folder = this Examples folder (for the load('private/...') calls).
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

if ~isempty(mfilename('fullpath')), cd(fileparts(mfilename('fullpath'))); end
addpath(genpath(fullfile('..','src')));
pause off

% ---- SISO Steps ----
try, Step_1_ExcitationDesign;                          savefigs('Step_1_ExcitationDesign');                          catch e, fprintf(2,'Step_1_ExcitationDesign: %s\n',e.message); end, close all
try, Step_2_NonparametricFRF;                          savefigs('Step_2_NonparametricFRF');                          catch e, fprintf(2,'Step_2_NonparametricFRF: %s\n',e.message); end, close all
try, Step_3_NonparametricFRF_LPM_thermal;             savefigs('Step_3_NonparametricFRF_LPM_thermal');             catch e, fprintf(2,'Step_3_thermal: %s\n',e.message); end, close all
try, Step_3_NonparametricFRF_LPM_positioning;         savefigs('Step_3_NonparametricFRF_LPM_positioning');         catch e, fprintf(2,'Step_3_positioning: %s\n',e.message); end, close all
try, Step_4_NonlinearDistortions;                      savefigs('Step_4_NonlinearDistortions');                      catch e, fprintf(2,'Step_4: %s\n',e.message); end, close all
try, Step_5_ParametricEstimation;                      savefigs('Step_5_ParametricEstimation');                      catch e, fprintf(2,'Step_5: %s\n',e.message); end, close all
try, Step_6_SelectionValidation;                       savefigs('Step_6_SelectionValidation');                       catch e, fprintf(2,'Step_6: %s\n',e.message); end, close all

% ---- MIMO Steps ----
try, Step_MIMO1_ExcitationDesign;                      savefigs('Step_MIMO1_ExcitationDesign');                      catch e, fprintf(2,'Step_MIMO1: %s\n',e.message); end, close all
try, Step_MIMO2_NonparametricFRF;                      savefigs('Step_MIMO2_NonparametricFRF');                      catch e, fprintf(2,'Step_MIMO2: %s\n',e.message); end, close all
try, Step_MIMO3_NonparametricFRF_LPM_positioning;     savefigs('Step_MIMO3_NonparametricFRF_LPM_positioning');     catch e, fprintf(2,'Step_MIMO3: %s\n',e.message); end, close all
try, Step_MIMO4_NonlinearDistortions;                  savefigs('Step_MIMO4_NonlinearDistortions');                  catch e, fprintf(2,'Step_MIMO4: %s\n',e.message); end, close all
try, Step_MIMO5_ParametricEstimation;                  savefigs('Step_MIMO5_ParametricEstimation');                  catch e, fprintf(2,'Step_MIMO5: %s\n',e.message); end, close all
try, Step_MIMO6_SelectionValidation;                   savefigs('Step_MIMO6_SelectionValidation');                   catch e, fprintf(2,'Step_MIMO6: %s\n',e.message); end, close all

% ---- SISO Tutorials ----
try, Tutorial_1_chirp;                                 savefigs('Tutorial_1_chirp');                                 catch e, fprintf(2,'Tutorial_1_chirp: %s\n',e.message); end, close all
try, Tutorial_1_qlog;                                  savefigs('Tutorial_1_qlog');                                  catch e, fprintf(2,'Tutorial_1_qlog: %s\n',e.message); end, close all
try, Tutorial_1_random;                                savefigs('Tutorial_1_random');                                catch e, fprintf(2,'Tutorial_1_random: %s\n',e.message); end, close all
try, Tutorial_2_iterative;                             savefigs('Tutorial_2_iterative');                             catch e, fprintf(2,'Tutorial_2_iterative: %s\n',e.message); end, close all
try, Tutorial_3_nonlinear_in;                          savefigs('Tutorial_3_nonlinear_in');                          catch e, fprintf(2,'Tutorial_3_nonlinear_in: %s\n',e.message); end, close all
try, Tutorial_3_nonlinear_out;                         savefigs('Tutorial_3_nonlinear_out');                         catch e, fprintf(2,'Tutorial_3_nonlinear_out: %s\n',e.message); end, close all

% ---- MIMO Tutorial ----
try, Tutorial_4_MIMO;                                  savefigs('Tutorial_4_MIMO');                                  catch e, fprintf(2,'Tutorial_4_MIMO: %s\n',e.message); end, close all

pause on
disp('=== all figures exported to Examples/plot/ ===');
