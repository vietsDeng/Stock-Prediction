-- phpMyAdmin SQL Dump
-- version 4.5.1
-- http://www.phpmyadmin.net
--
-- Host: 127.0.0.1
-- Generation Time: 2018-05-19 03:34:32
-- 服务器版本： 10.1.19-MariaDB
-- PHP Version: 5.6.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `prediction`
--

-- --------------------------------------------------------

--
-- 表的结构 `baidu_index`
--

CREATE TABLE `baidu_index` (
  `id` int(11) NOT NULL,
  `vocab_id` int(11) NOT NULL,
  `bindex` varchar(10000) NOT NULL,
  `date` date NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- 表的结构 `history`
--

CREATE TABLE `history` (
  `id` int(11) NOT NULL COMMENT '历史数据ID',
  `stock_id` int(11) DEFAULT NULL COMMENT '股票ID',
  `date` date DEFAULT NULL COMMENT '日期',
  `opening` decimal(12,2) NOT NULL DEFAULT '0.00' COMMENT '开盘价',
  `closing` decimal(12,2) NOT NULL DEFAULT '0.00' COMMENT '收盘价',
  `difference` decimal(12,2) NOT NULL DEFAULT '0.00' COMMENT '差价 (两天收盘价之差)',
  `percentage_difference` decimal(12,2) NOT NULL DEFAULT '0.00' COMMENT '差价百分比',
  `lowest` decimal(12,2) NOT NULL DEFAULT '0.00' COMMENT '最低价',
  `highest` decimal(12,2) NOT NULL DEFAULT '0.00' COMMENT '最高价',
  `volume` int(11) NOT NULL DEFAULT '0' COMMENT '成交量/手',
  `amount` decimal(20,2) NOT NULL DEFAULT '0.00' COMMENT '交易额/万元'
) ENGINE=MyISAM DEFAULT CHARSET=utf8 COMMENT='历史数据表';

-- --------------------------------------------------------

--
-- 表的结构 `news`
--

CREATE TABLE `news` (
  `id` int(11) NOT NULL,
  `title` varchar(256) NOT NULL,
  `content` varchar(10000) NOT NULL,
  `time` date DEFAULT NULL,
  `news_from` varchar(50) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8 COMMENT='财经新闻表';

-- --------------------------------------------------------

--
-- 表的结构 `stock`
--

CREATE TABLE `stock` (
  `id` int(11) NOT NULL COMMENT '股票ID',
  `name` varchar(100) NOT NULL COMMENT '股票名称',
  `code` varchar(30) NOT NULL COMMENT '股票代码'
) ENGINE=MyISAM DEFAULT CHARSET=utf8 COMMENT='股票表';

-- --------------------------------------------------------

--
-- 表的结构 `vocab`
--

CREATE TABLE `vocab` (
  `id` int(11) NOT NULL,
  `word` varchar(50) DEFAULT NULL,
  `baidu_code` int(11) DEFAULT NULL,
  `tfidf_ranking` int(11) DEFAULT NULL,
  `textrank_ranking` int(11) DEFAULT NULL,
  `status` bit(1) NOT NULL DEFAULT b'1'
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `baidu_index`
--
ALTER TABLE `baidu_index`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `history`
--
ALTER TABLE `history`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `news`
--
ALTER TABLE `news`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `stock`
--
ALTER TABLE `stock`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `vocab`
--
ALTER TABLE `vocab`
  ADD PRIMARY KEY (`id`);

--
-- 在导出的表使用AUTO_INCREMENT
--

--
-- 使用表AUTO_INCREMENT `baidu_index`
--
ALTER TABLE `baidu_index`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=318499;
--
-- 使用表AUTO_INCREMENT `history`
--
ALTER TABLE `history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '历史数据ID', AUTO_INCREMENT=24885;
--
-- 使用表AUTO_INCREMENT `news`
--
ALTER TABLE `news`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=765539;
--
-- 使用表AUTO_INCREMENT `stock`
--
ALTER TABLE `stock`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '股票ID', AUTO_INCREMENT=8;
--
-- 使用表AUTO_INCREMENT `vocab`
--
ALTER TABLE `vocab`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=121;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
