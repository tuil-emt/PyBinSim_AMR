import logging
import zmq
import numpy as np


class InlinePoseParser(object):
	def __init__(self, maxChannels):

		self.log = logging.getLogger("pybinsim.InlinePoseParser")
		self.log.info("InlinePoseParser: init")

		# Default values; Stores filter keys for all channels/convolvers
		self.filtersUpdated = [True] * maxChannels

		self.defaultValue = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
		self.valueList = [self.defaultValue] * maxChannels

	def parse_pose_input(self, channel, azi_lst, ele_lst, azi_src, ele_src):
		""" Compare new pose data with existing pose, determine if an update is needed """
		
		# we are just using first 2 elements of valueList for now
		# poseData = (azi_lst, ele_lst, 0, 0, 0, 0, azi_src, ele_src, 0, 0, 0, 0, 0, 0, 0)
		poseData = (azi_lst, ele_lst, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

		# compare poseData with valueList for channel
		if poseData != self.valueList[channel][:len(poseData)]:
			self.filtersUpdated[channel] = True
			self.valueList[channel] = poseData + self.defaultValue[len(poseData):]            

			# self.log.info("Channel: {}".format(str(channel)))
			# self.log.info("Args: {}".format(str(poseData)))

	def is_filter_update_necessary(self, channel):
		""" Check if there is a new filter for channel """
		return self.filtersUpdated[channel]

	def is_filter_update_necessary(self):
		""" Check if there is ANY new filter """
		return any(self.filtersUpdated)

	def get_current_values(self, channel):
		""" Return key for filter """
		self.filtersUpdated[channel] = False
		return self.valueList[channel]
